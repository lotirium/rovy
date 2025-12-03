#!/usr/bin/env python3
"""
Rovy Raspberry Pi - Robot Server with API
Provides:
1. REST API for mobile app (port 8000)
2. WebSocket client streaming to cloud PC (for AI)
3. Direct hardware control (rover, camera, audio)

Usage:
    python main_api.py
"""
import asyncio
import json
import time
import base64
import signal
import sys
import threading
import subprocess
from datetime import datetime
from typing import Optional
from io import BytesIO

import config

# FastAPI imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")

# Optional imports with fallbacks
try:
    import websockets
    WEBSOCKETS_OK = True
except ImportError:
    WEBSOCKETS_OK = False
    print("WARNING: websockets not installed. Cloud streaming disabled.")

try:
    import cv2
    CAMERA_OK = True
except ImportError:
    CAMERA_OK = False
    print("WARNING: OpenCV not installed. Camera disabled.")

try:
    import pyaudio
    import numpy as np
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    print("WARNING: PyAudio not installed. Microphone disabled.")

try:
    import sounddevice as sd
    import soundfile as sf
    PLAYBACK_OK = True
except ImportError:
    PLAYBACK_OK = False
    print("WARNING: sounddevice not installed. Audio playback disabled.")

try:
    from rover import Rover
    ROVER_OK = True
except Exception as e:
    ROVER_OK = False
    print(f"WARNING: Rover not available: {e}")


# ==============================================================================
# Pydantic Models for API
# ==============================================================================

class MoveCommand(BaseModel):
    direction: str
    distance: float = 0.5
    speed: str = "medium"

class HeadCommand(BaseModel):
    x: float
    y: float
    speed: int = 200

class LightCommand(BaseModel):
    front: int
    back: int

class NodCommand(BaseModel):
    times: int = 3

class StatusResponse(BaseModel):
    battery_voltage: Optional[float] = None
    battery_percent: Optional[int] = None
    temperature: Optional[float] = None
    rover_connected: bool = False
    camera_connected: bool = False

class HealthResponse(BaseModel):
    status: str
    version: str = "2.0"
    capabilities: dict

class ClaimRequestResponse(BaseModel):
    pin: str

class ClaimConfirmRequest(BaseModel):
    pin: str

class ClaimConfirmResponse(BaseModel):
    control_token: str
    robot_id: str

class ClaimControlResponse(BaseModel):
    session_id: str


# ==============================================================================
# Robot Client with API
# ==============================================================================

class RobotServer:
    """
    Robot server that provides both:
    1. REST API for mobile app
    2. WebSocket client for cloud streaming
    """
    
    def __init__(self):
        self.running = False
        self.ws = None  # Cloud WebSocket connection
        self.rover = None
        self.camera = None
        self.audio_stream = None
        
        # State
        self.is_listening = False
        self.audio_buffer = []
        self.last_image = None
        self.last_image_time = 0
        
        print("=" * 60)
        print("  ROVY ROBOT SERVER")
        print(f"  REST API: http://0.0.0.0:8000")
        print(f"  Cloud Stream: {config.SERVER_URL}")
        print("=" * 60)
    
    def init_rover(self):
        """Initialize rover connection."""
        if not ROVER_OK:
            print("[Rover] Not available")
            return False
        
        try:
            self.rover = Rover(config.ROVER_SERIAL_PORT, config.ROVER_BAUDRATE)
            self.rover.display_lines([
                "ROVY",
                "Starting...",
                "",
                ""
            ])
            print("[Rover] Connected")
            return True
        except Exception as e:
            print(f"[Rover] Connection failed: {e}")
            return False
    
    def init_camera(self):
        """Initialize camera."""
        if not CAMERA_OK:
            return False
        
        try:
            self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            ret, _ = self.camera.read()
            if ret:
                print("[Camera] Ready")
                return True
            else:
                print("[Camera] Failed to read frame")
                return False
        except Exception as e:
            print(f"[Camera] Init failed: {e}")
            return False
    
    def init_audio(self):
        """Initialize audio input (ReSpeaker)."""
        if not AUDIO_OK:
            return False
        
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Find ReSpeaker device
            device_index = None
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                name = info.get('name', '').lower()
                if 'respeaker' in name or 'seeed' in name:
                    device_index = i
                    print(f"[Audio] Found ReSpeaker: {info['name']}")
                    break
            
            self.audio_device_index = device_index
            print("[Audio] Ready")
            return True
        except Exception as e:
            print(f"[Audio] Init failed: {e}")
            return False
    
    def capture_image(self) -> Optional[bytes]:
        """Capture image from camera as JPEG bytes."""
        if not self.camera or not CAMERA_OK:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        self.last_image = buffer.tobytes()
        self.last_image_time = time.time()
        return self.last_image
    
    def get_cached_image(self) -> Optional[bytes]:
        """Get last captured image (for efficiency)."""
        # Cache images for 0.1 seconds to avoid re-capturing for multiple requests
        if self.last_image and (time.time() - self.last_image_time) < 0.1:
            return self.last_image
        return self.capture_image()
    
    async def connect_cloud(self):
        """Connect to cloud PC server via WebSocket."""
        if not WEBSOCKETS_OK:
            print("[Cloud] WebSocket not available, skipping cloud connection")
            return
        
        attempt = 0
        while self.running:
            attempt += 1
            try:
                print(f"[Cloud] Connecting to {config.SERVER_URL} (attempt {attempt})...")
                
                self.ws = await websockets.connect(
                    config.SERVER_URL,
                    ping_interval=30,
                    ping_timeout=10
                )
                
                print("[Cloud] Connected!")
                
                if self.rover:
                    self.rover.display_lines([
                        "ROVY",
                        "Cloud Connected",
                        config.PC_SERVER_IP,
                        ""
                    ])
                
                return True
                
            except Exception as e:
                print(f"[Cloud] Connection failed: {e}")
                
                if config.MAX_RECONNECT_ATTEMPTS > 0 and attempt >= config.MAX_RECONNECT_ATTEMPTS:
                    print("[Cloud] Max reconnect attempts reached, continuing without cloud")
                    return False
                
                await asyncio.sleep(config.RECONNECT_DELAY)
        
        return False
    
    async def stream_to_cloud(self):
        """Stream audio/video to cloud for AI processing."""
        if not self.ws:
            return
        
        print("[Cloud Stream] Starting...")
        
        image_interval = 1.0 / config.CAMERA_FPS
        sensor_interval = 5.0
        
        last_image_time = 0
        last_sensor_time = 0
        
        while self.running and self.ws:
            try:
                now = time.time()
                
                # Send image periodically
                if CAMERA_OK and self.camera and (now - last_image_time) >= image_interval:
                    image_bytes = self.capture_image()
                    if image_bytes:
                        await self.ws.send(json.dumps({
                            "type": "image_data",
                            "image_base64": base64.b64encode(image_bytes).decode('utf-8'),
                            "width": config.CAMERA_WIDTH,
                            "height": config.CAMERA_HEIGHT,
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                    last_image_time = now
                
                # Send sensor data periodically
                if self.rover and (now - last_sensor_time) >= sensor_interval:
                    status = self.rover.get_status()
                    if status:
                        await self.ws.send(json.dumps({
                            "type": "sensor_data",
                            "battery_voltage": status.get('voltage'),
                            "battery_percent": self.rover.voltage_to_percent(status.get('voltage')),
                            "temperature": status.get('temperature'),
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                    last_sensor_time = now
                
                await asyncio.sleep(0.01)
                
            except websockets.exceptions.ConnectionClosed:
                print("[Cloud Stream] Connection lost")
                break
            except Exception as e:
                print(f"[Cloud Stream] Error: {e}")
                await asyncio.sleep(0.1)
    
    async def receive_from_cloud(self):
        """Receive commands from cloud."""
        if not self.ws:
            return
        
        print("[Cloud Receive] Starting...")
        
        while self.running and self.ws:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                msg = json.loads(message)
                msg_type = msg.get('type', '')
                
                # Handle cloud commands (AI responses, etc.)
                if msg_type == 'speak':
                    text = msg.get('text', '')
                    print(f"[Cloud] Speak: {text[:50]}...")
                    # Could implement TTS here
                
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("[Cloud Receive] Connection lost")
                break
            except Exception as e:
                print(f"[Cloud Receive] Error: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.camera:
            self.camera.release()
        
        if hasattr(self, 'pyaudio') and self.pyaudio:
            self.pyaudio.terminate()
        
        if self.rover:
            self.rover.display_lines(["ROVY", "Shutdown", "", ""])
            self.rover.cleanup()
        
        print("[Server] Cleanup complete")


# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(title="ROVY Robot API", version="2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global robot server instance
robot_server: Optional[RobotServer] = None

# Claim state (for mobile app authentication)
import secrets
import hmac
import hashlib

CLAIM_STATE = {
    "claimed": False,
    "control_token_hash": None,
    "pin": None,
    "pin_exp": 0,
}

def hash_token(token: str) -> str:
    """Hash a token for secure storage."""
    return hashlib.sha256(token.encode()).hexdigest()

def verify_token(token: str) -> bool:
    """Verify a control token."""
    if not (CLAIM_STATE["claimed"] and CLAIM_STATE["control_token_hash"]):
        return False
    return hmac.compare_digest(hash_token(token), CLAIM_STATE["control_token_hash"])


def get_robot() -> RobotServer:
    """Get robot server instance."""
    if robot_server is None:
        raise HTTPException(status_code=503, detail="Robot server not initialized")
    return robot_server


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    robot = get_robot()
    return HealthResponse(
        status="ok",
        version="2.0",
        capabilities={
            "camera": CAMERA_OK and robot.camera is not None,
            "rover": ROVER_OK and robot.rover is not None,
            "audio": AUDIO_OK,
            "cloud_stream": WEBSOCKETS_OK
        }
    )


@app.get("/status")
async def get_status() -> StatusResponse:
    """Get robot status."""
    robot = get_robot()
    
    status_data = {
        "rover_connected": robot.rover is not None,
        "camera_connected": robot.camera is not None
    }
    
    if robot.rover:
        rover_status = robot.rover.get_status()
        if rover_status:
            voltage = rover_status.get('voltage')
            status_data["battery_voltage"] = voltage
            if voltage:
                status_data["battery_percent"] = robot.rover.voltage_to_percent(voltage)
            status_data["temperature"] = rover_status.get('temperature')
    
    return StatusResponse(**status_data)


@app.get("/shot")
async def get_shot():
    """Get single camera frame."""
    robot = get_robot()
    
    if not robot.camera:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    image_bytes = robot.capture_image()
    if not image_bytes:
        raise HTTPException(status_code=500, detail="Failed to capture image")
    
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/video")
async def video_stream():
    """MJPEG video stream."""
    robot = get_robot()
    
    if not robot.camera:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    def generate_frames():
        while True:
            image_bytes = robot.capture_image()
            if image_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
            time.sleep(1.0 / config.CAMERA_FPS)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/camera/ws")
async def camera_websocket(websocket: WebSocket):
    """WebSocket camera stream (base64 JPEG)."""
    robot = get_robot()
    
    if not robot.camera:
        await websocket.close(code=1011, reason="Camera not available")
        return
    
    await websocket.accept()
    print("[API] Camera WebSocket connected")
    
    try:
        while True:
            image_bytes = robot.get_cached_image()
            if image_bytes:
                await websocket.send_json({
                    "type": "frame",
                    "data": base64.b64encode(image_bytes).decode('utf-8'),
                    "timestamp": time.time()
                })
            await asyncio.sleep(1.0 / config.CAMERA_FPS)
    except WebSocketDisconnect:
        print("[API] Camera WebSocket disconnected")


@app.websocket("/json")
async def json_control_websocket(websocket: WebSocket):
    """WebSocket for real-time JSON motor/lights control."""
    robot = get_robot()
    await websocket.accept()
    print("[API] JSON control WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            cmd_type = data.get("T")
            
            if not robot.rover:
                continue
            
            # Motor control (T=1)
            if cmd_type == 1:
                left = data.get("L", 0)
                right = data.get("R", 0)
                robot.rover._send_direct(left, right)
            
            # Lights control (T=132)
            elif cmd_type == 132:
                front = data.get("IO4", 0)
                back = data.get("IO5", 0)
                robot.rover.lights_ctrl(front, back)
            
            # Gimbal control (T=133)
            elif cmd_type == 133:
                x = data.get("X", 0)
                y = data.get("Y", 0)
                speed = data.get("SPD", 200)
                acc = data.get("ACC", 10)
                robot.rover.gimbal_ctrl(x, y, speed, acc)
                
    except WebSocketDisconnect:
        print("[API] JSON control WebSocket disconnected")
        if robot.rover:
            robot.rover.stop()


@app.post("/control/move")
async def move_robot(command: MoveCommand):
    """Move the robot."""
    robot = get_robot()
    
    if not robot.rover:
        raise HTTPException(status_code=503, detail="Rover not available")
    
    def do_move():
        robot.rover.move(command.direction, command.distance, command.speed)
    
    threading.Thread(target=do_move, daemon=True).start()
    return {"status": "moving", **command.dict()}


@app.post("/control/stop")
async def stop_robot():
    """Stop the robot."""
    robot = get_robot()
    
    if not robot.rover:
        raise HTTPException(status_code=503, detail="Rover not available")
    
    robot.rover.stop()
    return {"status": "stopped"}


@app.post("/control/head")
async def move_head(command: HeadCommand):
    """Control gimbal/head position."""
    robot = get_robot()
    
    if not robot.rover:
        raise HTTPException(status_code=503, detail="Rover not available")
    
    robot.rover.gimbal_ctrl(command.x, command.y, command.speed, 10)
    return command


@app.post("/control/lights")
async def control_lights(command: LightCommand):
    """Control lights."""
    robot = get_robot()
    
    if not robot.rover:
        raise HTTPException(status_code=503, detail="Rover not available")
    
    robot.rover.lights_ctrl(command.front, command.back)
    return command


@app.post("/control/nod")
async def nod(command: NodCommand):
    """Make robot nod."""
    robot = get_robot()
    
    if not robot.rover:
        raise HTTPException(status_code=503, detail="Rover not available")
    
    def do_nod():
        robot.rover.nod_yes(command.times)
    
    threading.Thread(target=do_nod, daemon=True).start()
    return command


@app.get("/wifi/status")
async def get_wifi_status():
    """Get WiFi connection status."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE,STATE", "device", "status"],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_connected = "wifi:connected" in result.stdout
        
        return {
            "connected": is_connected,
            "ssid": "unknown"  # Could parse from nmcli
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.get("/mode")
async def get_mode():
    """Get robot operating mode."""
    return {"mode": "autonomous"}


@app.post("/claim/request")
async def claim_request() -> ClaimRequestResponse:
    """Generate a PIN code for claiming the robot."""
    robot = get_robot()
    
    # Generate 6-digit PIN
    CLAIM_STATE["pin"] = f"{secrets.randbelow(10**6):06d}"
    CLAIM_STATE["pin_exp"] = time.time() + 120  # 2 minutes
    
    # Display PIN on robot's OLED
    if robot.rover:
        robot.rover.display_lines([
            "CLAIM PIN:",
            CLAIM_STATE["pin"][:3] + " " + CLAIM_STATE["pin"][3:],
            "Enter in app",
            "120s timeout"
        ])
    
    print(f"[Claim] PIN generated: {CLAIM_STATE['pin']}")
    return ClaimRequestResponse(pin=CLAIM_STATE["pin"])


@app.post("/claim/confirm")
async def claim_confirm(request: ClaimConfirmRequest) -> ClaimConfirmResponse:
    """Confirm PIN and generate control token."""
    robot = get_robot()
    
    # Verify PIN
    if (request.pin != CLAIM_STATE["pin"] or 
        time.time() > CLAIM_STATE["pin_exp"] or 
        CLAIM_STATE["claimed"]):
        raise HTTPException(status_code=400, detail="invalid_or_expired_pin")
    
    # Generate control token
    token = secrets.token_urlsafe(32)
    CLAIM_STATE["control_token_hash"] = hash_token(token)
    CLAIM_STATE["claimed"] = True
    CLAIM_STATE["pin"] = None
    CLAIM_STATE["pin_exp"] = 0
    
    # Reset OLED display
    if robot.rover:
        robot.rover.display_lines([
            "ROVY",
            "Connected",
            "",
            ""
        ])
    
    print("[Claim] Robot claimed successfully")
    return ClaimConfirmResponse(
        control_token=token,
        robot_id="rovy-pi"
    )


@app.post("/claim/release")
async def claim_release():
    """Release the claim."""
    if not CLAIM_STATE["claimed"]:
        raise HTTPException(status_code=400, detail="not_claimed")
    
    CLAIM_STATE["claimed"] = False
    CLAIM_STATE["control_token_hash"] = None
    
    print("[Claim] Robot claim released")
    return {"released": True}


@app.post("/claim-control")
async def claim_control() -> ClaimControlResponse:
    """Claim control session."""
    # Generate session ID
    session_id = secrets.token_urlsafe(16)
    
    print(f"[Claim] Control session claimed: {session_id}")
    return ClaimControlResponse(session_id=session_id)


# ==============================================================================
# Main Server Startup
# ==============================================================================

async def run_robot_server():
    """Run the robot server (hardware + cloud streaming)."""
    global robot_server
    
    robot_server = RobotServer()
    robot_server.running = True
    
    # Initialize hardware
    robot_server.init_rover()
    robot_server.init_camera()
    robot_server.init_audio()
    
    # Try to connect to cloud (non-blocking)
    if WEBSOCKETS_OK:
        if await robot_server.connect_cloud():
            # Run cloud streaming in background
            asyncio.create_task(robot_server.stream_to_cloud())
            asyncio.create_task(robot_server.receive_from_cloud())
        else:
            print("[Cloud] Continuing without cloud connection")
    
    # Keep running
    try:
        while robot_server.running:
            await asyncio.sleep(1)
    except:
        robot_server.cleanup()


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    print("\n[Signal] Shutting down...")
    if robot_server:
        robot_server.running = False
        robot_server.cleanup()
    sys.exit(0)


def main():
    """Main entry point."""
    if not FASTAPI_OK:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        return
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n" + "=" * 60)
    print("  ROVY ROBOT SERVER - Starting")
    print("=" * 60)
    print("  REST API for mobile app: http://0.0.0.0:8000")
    print("  Cloud streaming: " + ("enabled" if WEBSOCKETS_OK else "disabled"))
    print("=" * 60 + "\n")
    
    # Configure uvicorn
    config_uvicorn = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        loop="asyncio"
    )
    
    server = uvicorn.Server(config_uvicorn)
    
    # Add robot server startup to uvicorn lifespan
    async def startup():
        asyncio.create_task(run_robot_server())
    
    app.add_event_handler("startup", startup)
    
    # Run server
    server.run()


if __name__ == "__main__":
    main()

