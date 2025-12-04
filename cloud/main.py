#!/usr/bin/env python3
"""
Rovy Cloud Server - Unified Entry Point
Runs both:
- FastAPI REST API (port 8000) - for mobile app
- WebSocket server (port 8765) - for robot communication

Usage:
    python main.py
"""
import asyncio
import json
import time
import base64
import signal
import sys
import logging
import threading
from datetime import datetime
from typing import Set, Optional

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('RovyCloud')

# Check dependencies
try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
    WEBSOCKETS_OK = True
except ImportError:
    WEBSOCKETS_OK = False
    logger.error("websockets not installed. Run: pip install websockets")

try:
    import uvicorn
    UVICORN_OK = True
except ImportError:
    UVICORN_OK = False
    logger.warning("uvicorn not installed. REST API disabled. Run: pip install uvicorn")

# Import AI modules
try:
    from ai import CloudAssistant
    AI_OK = True
except ImportError as e:
    AI_OK = False
    logger.warning(f"AI module not available: {e}")
    CloudAssistant = None

try:
    from speech import SpeechProcessor
    SPEECH_OK = True
except ImportError as e:
    SPEECH_OK = False
    logger.warning(f"Speech module not available: {e}")
    SpeechProcessor = None


class RobotConnection:
    """Manages WebSocket connection to the robot (Raspberry Pi)."""
    
    def __init__(self, server: 'RovyCloudServer'):
        self.server = server
        self.clients: Set[WebSocketServerProtocol] = set()
        self.last_image: Optional[bytes] = None
        self.last_sensors = {}
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str = "/"):
        """Handle robot WebSocket connection."""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"🤖 Robot connected: {client_addr}")
        
        self.clients.add(websocket)
        
        try:
            await self.send_speak(websocket, f"Connected to {config.ASSISTANT_NAME} cloud")
            
            async for message in websocket:
                try:
                    await self.handle_message(websocket, message)
                except Exception as e:
                    logger.error(f"Message error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Robot disconnected: {client_addr}")
        finally:
            self.clients.discard(websocket)
    
    async def handle_message(self, websocket: WebSocketServerProtocol, raw_message: str):
        """Process message from robot."""
        msg = json.loads(raw_message)
        msg_type = msg.get('type', '')
        
        if msg_type == 'audio_data':
            await self.handle_audio(websocket, msg)
        elif msg_type == 'image_data':
            await self.handle_image(websocket, msg)
        elif msg_type == 'text_query':
            await self.handle_text_query(websocket, msg)
        elif msg_type == 'sensor_data':
            self.handle_sensor_data(msg)
        elif msg_type == 'wake_word':
            await self.send_speak(websocket, "Yes? I'm listening.")
        elif msg_type == 'ping':
            await websocket.send(json.dumps({"type": "pong"}))
    
    async def handle_audio(self, websocket: WebSocketServerProtocol, msg: dict):
        """Process audio from robot microphone."""
        if not self.server.speech:
            return
        
        logger.info("🎤 Processing audio...")
        
        try:
            audio_bytes = base64.b64decode(msg.get('audio_base64', ''))
            sample_rate = msg.get('sample_rate', 16000)
            
            # Run Whisper STT
            text = await asyncio.get_event_loop().run_in_executor(
                None, self.server.speech.transcribe, audio_bytes, sample_rate
            )
            
            if text:
                logger.info(f"📝 Heard: '{text}'")
                
                # Check for wake word
                text_lower = text.lower()
                wake_detected = any(w in text_lower for w in config.WAKE_WORDS)
                
                if wake_detected:
                    query = text_lower
                    for wake in config.WAKE_WORDS:
                        query = query.replace(wake, "").strip()
                    
                    if query:
                        await self.process_query(websocket, query)
                    else:
                        await self.send_speak(websocket, "Yes? How can I help?")
                else:
                    await self.process_query(websocket, text)
                    
        except Exception as e:
            logger.error(f"Audio error: {e}")
    
    async def handle_image(self, websocket: WebSocketServerProtocol, msg: dict):
        """Store latest camera frame from robot."""
        try:
            img_data = msg.get('image_base64', '')
            if img_data:
                self.last_image = base64.b64decode(img_data)
                logger.debug(f"📷 Image received ({len(self.last_image)} bytes)")
            else:
                logger.warning("Empty image_base64 in message")
        except Exception as e:
            logger.error(f"Image error: {e}")
    
    async def handle_text_query(self, websocket: WebSocketServerProtocol, msg: dict):
        """Handle text query from robot."""
        text = msg.get('text', '')
        use_vision = msg.get('include_vision', False)
        logger.info(f"💬 Query: '{text}'")
        await self.process_query(websocket, text, use_vision=use_vision)
    
    def handle_sensor_data(self, msg: dict):
        """Store sensor readings from robot."""
        self.last_sensors = {
            'battery_voltage': msg.get('battery_voltage'),
            'battery_percent': msg.get('battery_percent'),
            'temperature': msg.get('temperature'),
            'imu': {
                'roll': msg.get('imu_roll'),
                'pitch': msg.get('imu_pitch'),
                'yaw': msg.get('imu_yaw')
            }
        }
    
    async def process_query(self, websocket: WebSocketServerProtocol, query: str, use_vision: bool = None):
        """Process query using AI models."""
        if not self.server.assistant:
            await self.send_speak(websocket, "AI not available")
            return
        
        logger.info(f"🎯 Processing: '{query}'")
        
        # Auto-detect vision need
        if use_vision is None:
            vision_keywords = ['see', 'look', 'what is', 'who is', 'describe', 'camera', 'front']
            use_vision = any(kw in query.lower() for kw in vision_keywords)
        
        # Log vision status
        if use_vision:
            if self.last_image:
                logger.info(f"👁️  Using vision (image: {len(self.last_image)} bytes)")
            else:
                logger.warning("👁️  Vision requested but no image available!")
        
        try:
            if use_vision and self.last_image:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.server.assistant.ask_with_vision, query, self.last_image
                )
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.server.assistant.ask, query
                )
            
            await self.send_speak(websocket, response)
            
            # Check for movement commands
            movement = self.server.assistant.extract_movement(response, query)
            if movement:
                await self.send_move(websocket, **movement)
                
        except Exception as e:
            logger.error(f"Query error: {e}")
            await self.send_speak(websocket, "Sorry, I had trouble with that.")
    
    # === Commands to robot ===
    
    async def send_speak(self, websocket: WebSocketServerProtocol, text: str):
        """Send TTS to robot."""
        audio_b64 = None
        if self.server.speech:
            try:
                audio_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self.server.speech.synthesize, text
                )
                if audio_bytes:
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            except Exception as e:
                logger.warning(f"TTS failed: {e}")
        
        msg = {"type": "speak", "text": text, "audio_base64": audio_b64}
        await websocket.send(json.dumps(msg))
        logger.info(f"🔊 Sent: '{text[:50]}...'")
    
    async def send_move(self, websocket: WebSocketServerProtocol,
                        direction: str, distance: float = 0.5, speed: str = "medium"):
        """Send movement command to robot."""
        msg = {"type": "move", "direction": direction, "distance": distance, "speed": speed}
        await websocket.send(json.dumps(msg))
        logger.info(f"🚗 Move: {direction} {distance}m")
    
    async def send_gimbal(self, websocket: WebSocketServerProtocol, pan: float, tilt: float):
        """Send gimbal command to robot."""
        msg = {"type": "gimbal", "pan": pan, "tilt": tilt, "action": "move"}
        await websocket.send(json.dumps(msg))
    
    async def broadcast(self, msg: dict):
        """Send to all connected robots."""
        for client in self.clients:
            try:
                await client.send(json.dumps(msg))
            except:
                pass


class RovyCloudServer:
    """
    Unified cloud server for Rovy robot.
    - REST API for mobile app (FastAPI on port 8000)
    - WebSocket for robot communication (port 8765)
    - AI processing (LLM, Vision, Speech)
    """
    
    def __init__(self):
        self.running = False
        self.assistant = None
        self.speech = None
        self.robot = RobotConnection(self)
        
        logger.info("=" * 60)
        logger.info("  ROVY CLOUD SERVER")
        logger.info(f"  REST API: http://0.0.0.0:{config.API_PORT}")
        logger.info(f"  WebSocket: ws://0.0.0.0:{config.WS_PORT}")
        logger.info(f"  Tailscale: {config.PC_SERVER_IP}")
        logger.info("=" * 60)
        
        # Initialize AI
        self._init_ai()
    
    def _init_ai(self):
        """Initialize AI models."""
        logger.info("Loading AI models...")
        
        if AI_OK and CloudAssistant:
            try:
                # Qwen2-VL - no parameters needed, uses defaults
                self.assistant = CloudAssistant()
                logger.info("✅ AI assistant ready (Qwen2-VL)")
            except Exception as e:
                logger.error(f"AI init failed: {e}")
        
        if SPEECH_OK and SpeechProcessor:
            try:
                self.speech = SpeechProcessor(
                    whisper_model=config.WHISPER_MODEL,
                    tts_engine=config.TTS_ENGINE,
                    piper_voices=config.PIPER_VOICES
                )
                logger.info("✅ Speech processor ready")
            except Exception as e:
                logger.error(f"Speech init failed: {e}")
    
    async def run_websocket_server(self):
        """Run WebSocket server for robot connection."""
        if not WEBSOCKETS_OK:
            logger.error("WebSocket server disabled - websockets not installed")
            return
        
        async with serve(
            self.robot.handle_connection,
            config.HOST,
            config.WS_PORT,
            ping_interval=30,
            ping_timeout=10
        ):
            logger.info(f"✅ WebSocket server running on ws://{config.HOST}:{config.WS_PORT}")
            while self.running:
                await asyncio.sleep(1)
    
    def run_api_server(self):
        """Run FastAPI REST server for mobile app."""
        if not UVICORN_OK:
            logger.error("REST API disabled - uvicorn not installed")
            return
        
        # Import FastAPI app
        try:
            from app.main import app
            logger.info(f"✅ REST API running on http://{config.HOST}:{config.API_PORT}")
            uvicorn.run(app, host=config.HOST, port=config.API_PORT, log_level="warning")
        except Exception as e:
            logger.error(f"REST API failed: {e}")
    
    async def start(self):
        """Start all servers."""
        self.running = True
        
        # Run REST API in separate thread
        if UVICORN_OK:
            api_thread = threading.Thread(target=self.run_api_server, daemon=True)
            api_thread.start()
        
        # Run WebSocket server in main async loop
        await self.run_websocket_server()
    
    def stop(self):
        """Stop all servers."""
        self.running = False
        logger.info("🛑 Server stopping...")


server: RovyCloudServer = None


def get_server() -> RovyCloudServer:
    """Get the global server instance (for use by FastAPI endpoints)."""
    return server


async def broadcast_to_robot(text: str):
    """Send a speak command to all connected robots."""
    if server and server.robot.clients:
        msg = {"type": "speak", "text": text}
        await server.robot.broadcast(msg)
        logger.info(f"🔊 Broadcast to robot: '{text[:50]}...'")
        return True
    return False


def signal_handler(sig, frame):
    logger.info("\n👋 Shutting down...")
    if server:
        server.stop()
    sys.exit(0)


async def main():
    global server
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    ROVY CLOUD SERVER                          ║
    ║              Unified AI + API + Robot Hub                     ║
    ║                                                               ║
    ║  Services:                                                    ║
    ║  • REST API (port 8000) - Mobile app connection              ║
    ║  • WebSocket (port 8765) - Robot connection                  ║
    ║  • AI: LLM + Vision + Speech (local models)                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server = RovyCloudServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())

