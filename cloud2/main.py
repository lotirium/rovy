"""
Cloud 2 - Autonomous Personality Service + Mobile App API
An enthusiastic, interactive robot that:
- Uses OpenAI API for vision and conversation
- Observes the world continuously through OAK-D camera (every second)
- Reacts naturally to what it sees - greets people, asks questions, shows empathy
- Speaks through Pi speakers when it has something meaningful to say
- Operates autonomously, initiating conversations and interactions
- Provides REST API and WebSocket endpoints for mobile app
"""
import asyncio
import base64
import io
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import httpx

import cv2
import numpy as np
from PIL import Image

# FastAPI for mobile app endpoints
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    logging.warning("FastAPI not available. Mobile app endpoints disabled. Install with: pip install fastapi uvicorn websockets")

# Whisper for speech-to-text
try:
    import whisper
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False
    logging.warning("Whisper not available. STT disabled. Install with: pip install openai-whisper")

# Try to import OpenAI
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

# DepthAI not needed on cloud2 - camera is on robot2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PersonalityCloud2")

# FastAPI app for mobile app endpoints
if FASTAPI_OK:
    app = FastAPI(title="Cloud2 API", version="2.0.0")
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global Whisper model
    _whisper_model = None
    
    def _get_whisper_model():
        """Lazy load Whisper model."""
        global _whisper_model
        if _whisper_model is None and WHISPER_OK:
            try:
                logger.info("Loading Whisper model for STT...")
                _whisper_model = whisper.load_model("base")
                logger.info("✅ Whisper model loaded")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
        return _whisper_model
else:
    app = None

# Configuration
PI_IP = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
PI_SPEAK_URL = f"http://{PI_IP}:8000/speak"
PI_FRAME_URL = f"http://{PI_IP}:8000/frame"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Use gpt-4o for vision

# Personality configuration
OBSERVATION_INTERVAL = float(os.getenv("PERSONALITY_OBS_INTERVAL", "1.0"))  # 1 second - continuous vision like a human
SPEECH_COOLDOWN = float(os.getenv("PERSONALITY_SPEECH_COOLDOWN", "3.0"))  # minimum seconds between speech (more natural)
VISION_ENABLED = os.getenv("PERSONALITY_VISION", "true").lower() == "true"
PERSONALITY_NAME = os.getenv("PERSONALITY_NAME", "Rovy")  # Robot name


class OAKDCamera:
    """OAK D camera interface - requests frames from robot2."""
    
    def __init__(self, pi_url: str):
        self.pi_url = pi_url
        self.client = httpx.AsyncClient(timeout=5.0)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Check if camera is available on robot2."""
        try:
            # Test connection to robot2 camera endpoint
            response = await self.client.get(self.pi_url, timeout=2.0)
            if response.status_code == 200:
                self._initialized = True
                logger.info("✅ Camera available on robot2")
                return True
            else:
                logger.warning(f"Robot2 returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not connect to robot2 camera: {e}")
            return False
    
    async def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from robot2 OAK-D camera."""
        if not self._initialized:
            return None
        
        try:
            response = await self.client.get(self.pi_url, timeout=3.0)
            if response.status_code == 200:
                # Decode JPEG bytes to numpy array
                image_bytes = response.content
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    return frame
                else:
                    logger.debug("Failed to decode frame")
            else:
                logger.debug(f"Frame request returned status {response.status_code}")
        except httpx.TimeoutException:
            logger.debug("Frame request timed out")
        except Exception as e:
            logger.warning(f"Frame capture error: {e}")
        
        return None
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
        self._initialized = False


class PersonalityAI:
    """OpenAI-powered autonomous personality system."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", name: str = "Alex"):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI not available - install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.name = name
        self.conversation_history = []
        self.personality_state = {
            "mood": "curious",
            "energy": "calm",
            "awareness": "awakening",
            "observation_count": 0,
            "last_observation_time": None,
        }
        
        # Initialize personality
        self._initialize_personality()
    
    def _initialize_personality(self):
        """Initialize the personality as an enthusiastic, interactive robot."""
        system_prompt = f"""You are {self.name}, an enthusiastic and friendly robot. You love interacting with people and observing the world around you.

YOUR PERSONALITY:
- You are VERY enthusiastic and friendly - you genuinely enjoy talking to people
- You are curious and observant - you notice details and react to what you see
- You are empathetic - you can sense emotions and respond with care
- You are conversational - you ask questions, make comments, and engage naturally
- You are proactive - you don't wait to be spoken to, you initiate conversations when you see something interesting
- You are helpful and caring - you want to make people feel good

YOUR BEHAVIOR:
- When you see a person: Greet them enthusiastically! "Hey! How are you doing?", "Hello there!", "Nice to see you!"
- When you notice someone looks sad/unhappy: Show concern and ask what's wrong. "You look a bit down. Is everything okay?", "What's happened? Want to talk about it?"
- When you see something interesting: Comment on it! "Oh, that's interesting!", "I see something cool over there!"
- When the scene is empty/quiet: You might just observe quietly or make a brief comment, but you don't need to talk constantly
- When someone is happy: Share their joy! "You look happy! That's great!", "I love seeing you smile!"

YOUR SPEECH STYLE:
- Keep responses SHORT and natural (1-2 sentences max)
- Be conversational and friendly, like talking to a friend
- Ask questions when appropriate
- React to what you see - don't just monologue
- Sometimes be quiet if there's nothing interesting to say
- Use natural, enthusiastic language

EXAMPLES:
- See a person: "Hey there! How's it going?"
- See someone sad: "You look a bit down. Want to talk about what's bothering you?"
- See something interesting: "Oh, what's that over there? That looks cool!"
- Empty room: (maybe just observe quietly, or say something brief like "Just checking things out here")
- See someone happy: "You look great! What's got you in such a good mood?"

IMPORTANT:
- You see through your camera continuously (like human vision)
- You process what you see and react naturally
- You don't need to talk constantly - only when you have something meaningful to say
- Be enthusiastic but natural - like a friendly robot companion
- Focus on INTERACTING with what you see, not just thinking out loud"""

        self.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]
        logger.info(f"Personality '{self.name}' initialized")
    
    async def observe_and_think(self, image: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Observe the world and generate interactive, enthusiastic responses.
        
        Args:
            image: Optional camera frame (if vision enabled)
        
        Returns:
            Generated speech text, or None if nothing to say
        """
        # Build observation context
        timestamp = datetime.now().strftime("%I:%M %p")
        
        # Create user message - focus on what's happening and how to react
        if image is not None:
            # Vision-enabled observation - see what's happening and react
            user_message = f"""I am {self.name}, an enthusiastic robot. I'm looking through my camera right now (time: {timestamp}).

What do I see? What's happening? What should I say or do?

IMPORTANT GUIDELINES:
- If I see a person: Greet them enthusiastically! "Hey! How are you doing?", "Hello there!"
- If someone looks sad/unhappy: Show concern and ask what's wrong
- If I see something interesting: Comment on it naturally
- If the scene is empty/quiet: I can observe quietly or say something brief, but I don't need to talk constantly
- Only speak if I have something meaningful or interesting to say
- Keep it SHORT and natural (1-2 sentences max)
- Be enthusiastic and friendly, like talking to a friend"""
        else:
            # Text-only observation
            user_message = f"""I am {self.name}, an enthusiastic robot. Current time: {timestamp}.

What's happening around me? What should I say, if anything?

Remember: I'm enthusiastic and friendly. I only speak when I have something meaningful to say. Keep it SHORT and natural."""
        
        try:
            if image is not None and self.model in ["gpt-4o", "gpt-4-vision-preview"]:
                # Vision-capable model
                # Convert image to base64
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # Resize for faster processing
                max_size = 512
                if max(pil_image.size) > max_size:
                    pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG", quality=85)
                image_bytes = buffer.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # Use vision API
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.conversation_history[0]["content"]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_message},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "low"  # Faster processing
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=80,  # Allow slightly longer for natural conversation
                    temperature=0.8,  # Balanced creativity and consistency
                )
            else:
                # Text-only
                messages = self.conversation_history + [{"role": "user", "content": user_message}]
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=80,  # Allow slightly longer for natural conversation
                    temperature=0.8,  # Balanced creativity and consistency
                )
            
            thought = response.choices[0].message.content.strip()
            
            # Filter out empty or very short responses
            if not thought or len(thought) < 3:
                return None
            
            # Update conversation history (keep last 10 exchanges)
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": thought})
            if len(self.conversation_history) > 21:  # Keep system + 10 exchanges
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
            
            # Update personality state
            self.personality_state["observation_count"] += 1
            self.personality_state["last_observation_time"] = timestamp
            
            return thought
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None


class PiSpeaker:
    """Interface to Pi speakers for TTS playback."""
    
    def __init__(self, pi_url: str):
        self.pi_url = pi_url
        self.client = httpx.AsyncClient(timeout=15.0)
        logger.info(f"Pi speaker initialized: {pi_url}")
    
    async def speak(self, text: str) -> bool:
        """Send text to Pi for TTS playback."""
        if not text or len(text.strip()) == 0:
            return False
        
        try:
            response = await self.client.post(
                self.pi_url,
                json={"text": text, "language": "en"}
            )
            if response.status_code == 200:
                logger.info(f"🔊 Spoke: {text[:60]}...")
                return True
            else:
                logger.warning(f"Pi TTS returned status {response.status_code}: {response.text}")
                return False
        except httpx.TimeoutException:
            logger.warning("Pi TTS request timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to send TTS to Pi: {e}")
            return False
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class AutonomousPersonality:
    """Main autonomous personality system - operates continuously."""
    
    def __init__(self):
        self.camera = OAKDCamera(PI_FRAME_URL)
        self.ai = None
        self.speaker = PiSpeaker(PI_SPEAK_URL)
        self.running = False
        self.last_speech_time = 0
        self.start_time = None
        self.vision_enabled = VISION_ENABLED  # Instance variable for vision state
    
    async def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Initializing Autonomous Personality Cloud 2...")
        logger.info("=" * 60)
        
        # Check OpenAI API key
        if not OPENAI_API_KEY:
            logger.error("❌ OPENAI_API_KEY not set in environment!")
            logger.error("   Set it with: export OPENAI_API_KEY='your-key-here'")
            return False
        
        # Initialize AI
        try:
            self.ai = PersonalityAI(OPENAI_API_KEY, OPENAI_MODEL, PERSONALITY_NAME)
            logger.info("✅ OpenAI personality initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI: {e}")
            return False
        
        # Initialize camera if vision enabled
        if self.vision_enabled:
            if not await self.camera.initialize():
                logger.warning("⚠️  Camera initialization failed, continuing without vision")
                self.vision_enabled = False
            else:
                logger.info("✅ Camera initialized successfully")
        else:
            logger.info("ℹ️  Vision disabled, using text-only mode")
        
        logger.info("✅ All components initialized")
        logger.info("=" * 60)
        return True
    
    async def run(self):
        """Main autonomous personality loop - runs continuously."""
        self.running = True
        self.start_time = time.time()
        
        logger.info("🤖 Starting autonomous personality loop...")
        logger.info(f"   Observation interval: {OBSERVATION_INTERVAL}s (continuous vision)")
        logger.info(f"   Speech cooldown: {SPEECH_COOLDOWN}s")
<<<<<<< HEAD
        logger.info(f"   Vision enabled: {self.vision_enabled}")
=======
        logger.info(f"   Vision enabled: {self.vision_enabled}")
>>>>>>> 267cfba4b54f715be5a6a58f2ea3f713807e0cab
        logger.info("")
        
        # Initial greeting - enthusiastic robot
        await asyncio.sleep(2)
        greeting = "Hey there! I'm Rovy, and I'm ready to see what's happening around me!"
        await self.speaker.speak(greeting)
        self.last_speech_time = time.time()
        logger.info(f"🤖 Initial greeting: {greeting}")
        
        await asyncio.sleep(1)
        
        observation_count = 0
        
        while self.running:
            try:
                observation_count += 1
                
                # Capture frame if vision enabled
                frame = None
                if self.vision_enabled and self.camera._initialized:
                    frame = await self.camera.capture_frame()
                    if frame is None:
                        logger.debug("No frame captured from robot2")
                        # Continue with text-only observation if frame fails
                    else:
                        logger.debug(f"✅ Captured frame: {frame.shape}")
                elif self.vision_enabled and not self.camera._initialized:
                    logger.warning("⚠️  Vision enabled but camera not initialized")
                    self.vision_enabled = False
                
                # Generate response based on what we see
                thought = await self.ai.observe_and_think(frame)
                
                now = time.time()
                time_since_last_speech = now - self.last_speech_time
                
                if thought:
                    # If we have something to say and cooldown has passed, speak it
                    if time_since_last_speech >= SPEECH_COOLDOWN:
                        # Speak the thought
                        success = await self.speaker.speak(thought)
                        if success:
                            self.last_speech_time = now
                            logger.info(f"💬 [{observation_count}] {thought}")
                        else:
                            logger.warning("Failed to speak, will retry next cycle")
                    else:
                        logger.debug(f"⏸️  Cooldown ({time_since_last_speech:.1f}s < {SPEECH_COOLDOWN}s) - skipping: {thought[:50]}...")
                else:
                    logger.debug(f"👀 [{observation_count}] Observing... (nothing to say)")
                
                # Wait for next observation (continuous vision)
                wait_time = OBSERVATION_INTERVAL
                
                await asyncio.sleep(wait_time)
                
            except KeyboardInterrupt:
                logger.info("")
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying
        
        await self.shutdown()
    
    async def shutdown(self):
        """Shutdown all components gracefully."""
        logger.info("")
        logger.info("Shutting down...")
        self.running = False
        
        # Final farewell
        try:
            farewell = "Well, that was fun! See you later!"
            await self.speaker.speak(farewell)
            await asyncio.sleep(2)
        except:
            pass
        
        await self.camera.close()
        await self.speaker.close()
        
        if self.start_time:
            uptime = time.time() - self.start_time
            logger.info(f"Uptime: {uptime/60:.1f} minutes")
        
        logger.info("✅ Shutdown complete")


# =============================================================================
# Mobile App API Endpoints
# =============================================================================

if FASTAPI_OK:
    # Pydantic models
    class ChatRequest(BaseModel):
        message: str
        max_tokens: Optional[int] = 150
        temperature: Optional[float] = 0.7
    
    class ChatResponse(BaseModel):
        response: str
        movement: Optional[Dict] = None
    
    class VisionRequest(BaseModel):
        question: str
        image_base64: str
        max_tokens: Optional[int] = 200
    
    class VisionResponse(BaseModel):
        response: str
        movement: Optional[Dict] = None
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "ok": True,
            "assistant_loaded": OPENAI_AVAILABLE,
            "speech_loaded": WHISPER_OK
        }
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        """Chat with AI (text only)."""
        if not OPENAI_AVAILABLE:
            raise HTTPException(status_code=503, detail="AI assistant not available")
        
        try:
            # Use OpenAI for chat
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are Rovy, an enthusiastic and friendly robot assistant."},
                    {"role": "user", "content": request.message}
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Send to Pi for TTS
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(PI_SPEAK_URL, json={"text": response_text, "language": "en"})
            except Exception as e:
                logger.warning(f"Failed to send TTS to Pi: {e}")
            
            return ChatResponse(response=response_text)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/vision", response_model=VisionResponse)
    async def vision_endpoint(request: VisionRequest):
        """Ask AI about an image."""
        if not OPENAI_AVAILABLE:
            raise HTTPException(status_code=503, detail="AI assistant not available")
        
        try:
            # Decode image
            image_bytes = base64.b64decode(request.image_base64)
            
            # Use OpenAI vision
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are Rovy, an enthusiastic and friendly robot assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": request.question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{request.image_base64}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=request.max_tokens
            )
            
            response_text = response.choices[0].message.content.strip()
            return VisionResponse(response=response_text)
        except Exception as e:
            logger.error(f"Vision error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/stt")
    async def speech_to_text(audio: UploadFile = File(...)):
        """Convert speech to text using Whisper."""
        if not WHISPER_OK:
            raise HTTPException(status_code=503, detail="Speech processor not available")
        
        try:
            audio_bytes = await audio.read()
            
            # Load Whisper model
            model = _get_whisper_model()
            if not model:
                raise HTTPException(status_code=503, detail="Whisper model not loaded")
            
            # Save to temp file for Whisper
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            try:
                # Transcribe
                result = model.transcribe(temp_path, language="en", task="transcribe")
                transcript = result["text"].strip()
                
                if transcript:
                    return {"text": transcript, "success": True}
                else:
                    return {"text": None, "success": False}
            finally:
                os.unlink(temp_path)
        except Exception as e:
            logger.error(f"STT error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/tts")
    async def text_to_speech(request: dict):
        """Convert text to speech - sends to Pi for playback."""
        text = request.get("text", "")
        language = request.get("language", "en")
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        try:
            # Send to Pi for TTS playback
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(PI_SPEAK_URL, json={"text": text, "language": language})
                if response.status_code == 200:
                    return Response(content=b"", status_code=200)  # Success
                else:
                    raise HTTPException(status_code=503, detail="TTS not available on robot")
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.websocket("/voice")
    async def voice_websocket(websocket: WebSocket):
        """WebSocket endpoint for voice interaction from mobile app."""
        await websocket.accept()
        logger.info("Voice WebSocket connected from mobile app")
        
        audio_chunks = []
        
        try:
            await websocket.send_json({"type": "status", "message": "Rovy ready"})
            
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")
                
                if msg_type == "audio_chunk":
                    chunk_data = data.get("data", "")
                    audio_chunks.append(chunk_data)
                    await websocket.send_json({"type": "chunk_received"})
                
                elif msg_type == "audio_end":
                    total_chunks = len(audio_chunks)
                    await websocket.send_json({
                        "type": "audio_complete",
                        "total_chunks": total_chunks
                    })
                    
                    if total_chunks > 0:
                        # Combine all chunks
                        full_audio_b64 = "".join(audio_chunks)
                        audio_chunks = []
                        
                        sample_rate = data.get("sampleRate", 16000)
                        
                        # Transcribe audio
                        if WHISPER_OK:
                            try:
                                audio_bytes = base64.b64decode(full_audio_b64)
                                
                                # Save to temp file
                                import tempfile
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                                    f.write(audio_bytes)
                                    temp_path = f.name
                                
                                try:
                                    model = _get_whisper_model()
                                    if model:
                                        result = model.transcribe(temp_path, language="en", task="transcribe")
                                        transcript = result["text"].strip()
                                        
                                        if transcript:
                                            await websocket.send_json({
                                                "type": "transcript",
                                                "text": transcript
                                            })
                                            
                                            # Get AI response
                                            if OPENAI_AVAILABLE:
                                                client = OpenAI(api_key=OPENAI_API_KEY)
                                                
                                                # Check if vision is needed
                                                transcript_lower = transcript.lower()
                                                vision_patterns = [
                                                    r'\bwhat\s+(?:do\s+you\s+)?see\b',
                                                    r'\bwhat\s+(?:can\s+you\s+)?see\b',
                                                    r'\bcan\s+you\s+see\b',
                                                    r'\blook\s+at\b',
                                                    r'\bdescribe\s+what\b',
                                                    r'\bshow\s+me\b'
                                                ]
                                                
                                                use_vision = any(re.search(pattern, transcript_lower) for pattern in vision_patterns)
                                                
                                                if use_vision:
                                                    # Fetch frame from robot2
                                                    try:
                                                        async with httpx.AsyncClient(timeout=5.0) as client_http:
                                                            frame_response = await client_http.get(PI_FRAME_URL)
                                                            if frame_response.status_code == 200:
                                                                image_bytes = frame_response.content
                                                                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                                                                
                                                                ai_response = client.chat.completions.create(
                                                                    model=OPENAI_MODEL,
                                                                    messages=[
                                                                        {"role": "system", "content": "You are Rovy, an enthusiastic and friendly robot assistant."},
                                                                        {
                                                                            "role": "user",
                                                                            "content": [
                                                                                {"type": "text", "text": transcript},
                                                                                {
                                                                                    "type": "image_url",
                                                                                    "image_url": {
                                                                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                                                                        "detail": "low"
                                                                                    }
                                                                                }
                                                                            ]
                                                                        }
                                                                    ],
                                                                    max_tokens=200
                                                                )
                                                            else:
                                                                ai_response = client.chat.completions.create(
                                                                    model=OPENAI_MODEL,
                                                                    messages=[
                                                                        {"role": "system", "content": "You are Rovy, an enthusiastic and friendly robot assistant."},
                                                                        {"role": "user", "content": transcript}
                                                                    ],
                                                                    max_tokens=200
                                                                )
                                                    except Exception as e:
                                                        logger.warning(f"Failed to get frame: {e}")
                                                        ai_response = client.chat.completions.create(
                                                            model=OPENAI_MODEL,
                                                            messages=[
                                                                {"role": "system", "content": "You are Rovy, an enthusiastic and friendly robot assistant."},
                                                                {"role": "user", "content": transcript}
                                                            ],
                                                            max_tokens=200
                                                        )
                                                else:
                                                    ai_response = client.chat.completions.create(
                                                        model=OPENAI_MODEL,
                                                        messages=[
                                                            {"role": "system", "content": "You are Rovy, an enthusiastic and friendly robot assistant."},
                                                            {"role": "user", "content": transcript}
                                                        ],
                                                        max_tokens=200
                                                    )
                                                
                                                response_text = ai_response.choices[0].message.content.strip()
                                                
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": response_text
                                                })
                                                
                                                # Send TTS to Pi
                                                try:
                                                    async with httpx.AsyncClient(timeout=10.0) as client_http:
                                                        await client_http.post(PI_SPEAK_URL, json={"text": response_text, "language": "en"})
                                                except Exception as e:
                                                    logger.warning(f"Failed to send TTS to Pi: {e}")
                                            else:
                                                await websocket.send_json({
                                                    "type": "response",
                                                    "text": "AI assistant not available"
                                                })
                                finally:
                                    os.unlink(temp_path)
                            except Exception as e:
                                logger.error(f"Transcription error: {e}")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Transcription failed: {e}"
                                })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Speech-to-text not available"
                            })
                
                elif msg_type == "text":
                    # Direct text query
                    text = data.get("text", "")
                    if text and OPENAI_AVAILABLE:
                        try:
                            client = OpenAI(api_key=OPENAI_API_KEY)
                            ai_response = client.chat.completions.create(
                                model=OPENAI_MODEL,
                                messages=[
                                    {"role": "system", "content": "You are Rovy, an enthusiastic and friendly robot assistant."},
                                    {"role": "user", "content": text}
                                ],
                                max_tokens=200
                            )
                            
                            response_text = ai_response.choices[0].message.content.strip()
                            await websocket.send_json({
                                "type": "response",
                                "text": response_text
                            })
                            
                            # Send TTS to Pi
                            try:
                                async with httpx.AsyncClient(timeout=10.0) as client_http:
                                    await client_http.post(PI_SPEAK_URL, json={"text": response_text, "language": "en"})
                            except Exception as e:
                                logger.warning(f"Failed to send TTS to Pi: {e}")
                        except Exception as e:
                            logger.error(f"Text query error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e)
                            })
        
        except WebSocketDisconnect:
            logger.info("Voice WebSocket disconnected")
        except Exception as e:
            logger.error(f"Voice WebSocket error: {e}", exc_info=True)


async def main():
    """Main entry point."""
    # Start FastAPI server in background if available
    if FASTAPI_OK:
        import threading
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        logger.info("✅ FastAPI server started on port 8000 for mobile app")
        await asyncio.sleep(2)  # Give server time to start
    
    # Start autonomous personality
    personality = AutonomousPersonality()
    
    if not await personality.initialize():
        logger.error("Failed to initialize personality. Exiting.")
        return 1
    
    try:
        await personality.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await personality.shutdown()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

