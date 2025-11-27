"""
Rovy Cloud Server - Main Entry Point
Handles WebSocket connections and processes AI/vision/speech tasks
Uses LOCAL models (llama.cpp, Whisper, Piper) - no cloud APIs needed!
"""
import asyncio
import json
import logging
import sys
import signal
import base64
from typing import Dict, Set, Optional
from pathlib import Path

# Add parent to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets
from websockets.server import serve, WebSocketServerProtocol

from shared.config import server_config, assistant_config
from shared.messages import (
    parse_message, MessageType, BaseMessage,
    AudioMessage, ImageMessage, TextQueryMessage, SensorMessage,
    SpeakMessage, MoveMessage, GimbalMessage, DisplayMessage, ErrorMessage
)
from assistant import CloudAssistant
from speech import SpeechProcessor
from vision import VisionProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('RovyServer')


class RovyCloudServer:
    """
    Main cloud server handling AI processing using LOCAL models.
    - Text: Gemma/Llama via llama.cpp
    - Vision: LLaVA/Phi-3-Vision via llama.cpp
    - Speech: Whisper for STT, Piper for TTS
    """
    
    def __init__(self):
        self.config = server_config
        self.clients: Set[WebSocketServerProtocol] = set()
        self.client_info: Dict[WebSocketServerProtocol, dict] = {}
        
        # Initialize processors with local models
        logger.info("🚀 Initializing Rovy Cloud Server with LOCAL models...")
        
        self.assistant = CloudAssistant(
            text_model_path=self.config.local_llm_path,
            vision_model_path=self.config.local_vision_model_path,
            lazy_load_vision=True  # Load vision model on first use
        )
        
        self.speech = SpeechProcessor(
            whisper_model=self.config.whisper_model
        )
        
        self.vision = VisionProcessor(
            known_faces_dir=self.config.known_faces_dir,
            tolerance=self.config.face_recognition_tolerance
        )
        
        # State
        self.running = False
        self.last_image: Optional[bytes] = None
        self.last_sensors: Optional[SensorMessage] = None
        
        logger.info("✅ All processors initialized with LOCAL models")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str = "/"):
        """Handle a new client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🔗 New connection from {client_id}")
        
        self.clients.add(websocket)
        self.client_info[websocket] = {
            'id': client_id,
            'connected_at': asyncio.get_event_loop().time(),
            'last_seen': asyncio.get_event_loop().time()
        }
        
        try:
            # Send welcome message
            await self.send_speak(websocket, f"Connected to {assistant_config.name} server")
            
            async for message in websocket:
                try:
                    await self.handle_message(websocket, message)
                    self.client_info[websocket]['last_seen'] = asyncio.get_event_loop().time()
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    import traceback
                    traceback.print_exc()
                    await self.send_error(websocket, str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client {client_id} disconnected")
        finally:
            self.clients.discard(websocket)
            self.client_info.pop(websocket, None)
    
    async def handle_message(self, websocket: WebSocketServerProtocol, raw_message: str):
        """Process incoming message"""
        msg = parse_message(raw_message)
        
        if msg.type == MessageType.AUDIO_DATA.value:
            await self.handle_audio(websocket, msg)
        
        elif msg.type == MessageType.IMAGE_DATA.value:
            await self.handle_image(websocket, msg)
        
        elif msg.type == MessageType.TEXT_QUERY.value:
            await self.handle_text_query(websocket, msg)
        
        elif msg.type == MessageType.SENSOR_DATA.value:
            await self.handle_sensor_data(websocket, msg)
        
        elif msg.type == MessageType.WAKE_WORD_DETECTED.value:
            await self.handle_wake_word(websocket)
        
        elif msg.type == MessageType.PING.value:
            await websocket.send(BaseMessage(type="pong").to_json())
    
    async def handle_audio(self, websocket: WebSocketServerProtocol, msg: AudioMessage):
        """Process audio data - speech recognition using LOCAL Whisper"""
        logger.info("🎤 Processing audio with local Whisper...")
        
        try:
            audio_bytes = msg.get_audio_bytes()
            
            # Transcribe speech using local Whisper
            text = await asyncio.get_event_loop().run_in_executor(
                None, self.speech.transcribe, audio_bytes, msg.sample_rate
            )
            
            if text:
                logger.info(f"📝 Transcribed: '{text}'")
                
                # Check for wake word
                wake_detected = any(w in text.lower() for w in assistant_config.wake_words)
                
                if wake_detected:
                    # Remove wake word from query
                    query = text.lower()
                    for wake_word in assistant_config.wake_words:
                        query = query.replace(wake_word, "").strip()
                    
                    if query:
                        await self.process_query(websocket, query)
                    else:
                        await self.send_speak(websocket, "Yes? How can I help you?")
                else:
                    # Process as query anyway if no wake word mode
                    await self.process_query(websocket, text)
            else:
                logger.debug("No speech detected")
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            await self.send_error(websocket, f"Audio processing failed: {e}")
    
    async def handle_image(self, websocket: WebSocketServerProtocol, msg: ImageMessage):
        """Process image data - vision tasks using LOCAL models"""
        logger.info("📷 Processing image...")
        
        try:
            image_bytes = msg.get_image_bytes()
            self.last_image = image_bytes
            
            # Check for faces
            faces = await asyncio.get_event_loop().run_in_executor(
                None, self.vision.recognize_faces, image_bytes
            )
            
            if faces:
                recognized = [f for f in faces if f['name'] != "Unknown"]
                if recognized:
                    names = ", ".join([f['name'] for f in recognized])
                    await self.send_speak(websocket, f"I see {names}!")
            
            # Process vision query if provided
            if msg.query:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.assistant.ask_with_vision, msg.query, image_bytes
                )
                await self.send_speak(websocket, response)
                
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            await self.send_error(websocket, f"Vision processing failed: {e}")
    
    async def handle_text_query(self, websocket: WebSocketServerProtocol, msg: TextQueryMessage):
        """Handle text-based query using LOCAL LLM"""
        logger.info(f"💬 Query: '{msg.text}'")
        
        try:
            if msg.include_vision and self.last_image:
                # Vision query using local LLaVA/Phi-3
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.assistant.ask_with_vision, msg.text, self.last_image
                )
            else:
                # Text query using local Gemma/Llama
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.assistant.ask, msg.text
                )
            
            await self.send_speak(websocket, response)
            
            # Check for movement commands
            movement = self.assistant.extract_movement(response, msg.text)
            if movement:
                await self.send_move(websocket, **movement)
                
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            await self.send_speak(websocket, "Sorry, I had trouble with that.")
    
    async def handle_sensor_data(self, websocket: WebSocketServerProtocol, msg: SensorMessage):
        """Store sensor data"""
        self.last_sensors = msg
        logger.debug(f"📊 Sensors: battery={msg.battery_percent}%")
    
    async def handle_wake_word(self, websocket: WebSocketServerProtocol):
        """Handle wake word detection"""
        logger.info("👋 Wake word detected!")
        await self.send_speak(websocket, "Yes? I'm listening.")
    
    async def process_query(self, websocket: WebSocketServerProtocol, query: str):
        """Process a voice/text query"""
        logger.info(f"🎯 Processing: '{query}'")
        
        # Check for vision-related queries
        vision_keywords = ['see', 'look', 'what is', 'who is', 'describe', 'show', 'in front', 'camera']
        needs_vision = any(kw in query.lower() for kw in vision_keywords)
        
        if needs_vision and self.last_image:
            # Use local vision model
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.assistant.ask_with_vision, query, self.last_image
            )
        else:
            # Use local text model
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.assistant.ask, query
            )
        
        await self.send_speak(websocket, response)
        
        # Check for movement commands
        movement = self.assistant.extract_movement(response, query)
        if movement:
            await self.send_move(websocket, **movement)
    
    # === Send Commands ===
    
    async def send_speak(self, websocket: WebSocketServerProtocol, text: str):
        """Send TTS command with pre-synthesized audio using LOCAL Piper/espeak"""
        # Generate audio using local TTS
        audio_base64 = None
        try:
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None, self.speech.synthesize, text
            )
            if audio_bytes:
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            logger.warning(f"Local TTS synthesis failed: {e}")
        
        msg = SpeakMessage(text=text, audio_base64=audio_base64)
        await websocket.send(msg.to_json())
        logger.info(f"🔊 Sent: '{text[:50]}...'")
    
    async def send_move(self, websocket: WebSocketServerProtocol, 
                        direction: str, distance: float = 0.5, speed: str = "medium"):
        """Send movement command"""
        msg = MoveMessage(direction=direction, distance=distance, speed=speed)
        await websocket.send(msg.to_json())
        logger.info(f"🚗 Move: {direction} {distance}m at {speed}")
    
    async def send_gimbal(self, websocket: WebSocketServerProtocol,
                          pan: float, tilt: float, speed: int = 200):
        """Send gimbal command"""
        msg = GimbalMessage(pan=pan, tilt=tilt, speed=speed)
        await websocket.send(msg.to_json())
    
    async def send_display(self, websocket: WebSocketServerProtocol, lines: list):
        """Send display update"""
        msg = DisplayMessage(lines=lines[:4])
        await websocket.send(msg.to_json())
    
    async def send_error(self, websocket: WebSocketServerProtocol, error: str):
        """Send error message"""
        msg = ErrorMessage(error=error)
        await websocket.send(msg.to_json())
    
    async def broadcast(self, message: BaseMessage):
        """Send to all clients"""
        if self.clients:
            await asyncio.gather(
                *[client.send(message.to_json()) for client in self.clients],
                return_exceptions=True
            )
    
    async def start(self):
        """Start the WebSocket server"""
        self.running = True
        
        logger.info(f"🌐 Starting server on ws://{self.config.host}:{self.config.port}")
        
        async with serve(
            self.handle_connection,
            self.config.host,
            self.config.port,
            ping_interval=30,
            ping_timeout=10
        ):
            logger.info(f"✅ Server running on ws://{self.config.host}:{self.config.port}")
            logger.info("Using LOCAL models (llama.cpp, Whisper, Piper)")
            logger.info("Waiting for connections...")
            
            while self.running:
                await asyncio.sleep(1)
    
    def stop(self):
        """Stop the server"""
        self.running = False
        logger.info("🛑 Server stopping...")


# Global server instance
server: RovyCloudServer = None


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    logger.info("\n👋 Shutting down...")
    if server:
        server.stop()
    sys.exit(0)


async def main():
    global server
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║              ROVY CLOUD SERVER                            ║
    ║         AI-Powered Robot Assistant                        ║
    ║                                                           ║
    ║  Using LOCAL models:                                      ║
    ║  • Text: Gemma/Llama via llama.cpp                       ║
    ║  • Vision: LLaVA/Phi-3-Vision via llama.cpp              ║
    ║  • Speech: Whisper (STT) + Piper (TTS)                   ║
    ║  • Face Recognition: dlib CNN                            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start server
    server = RovyCloudServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
