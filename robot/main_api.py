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

class VolumeCommand(BaseModel):
    volume: int  # 0-100


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
        """Initialize camera - automatically find USB camera."""
        if not CAMERA_OK:
            return False
        
        # Try indices in order: 1 (USB Camera), 0, 2
        for camera_index in [1, 0, 2]:
            try:
                print(f"[Camera] Trying /dev/video{camera_index}...")
                self.camera = cv2.VideoCapture(camera_index)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    print(f"[Camera] Ready on /dev/video{camera_index}")
                    return True
                else:
                    self.camera.release()
                    self.camera = None
            except Exception as e:
                if self.camera:
                    self.camera.release()
                    self.camera = None
        
        print("[Camera] No working camera found")
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
    
    def init_volume(self):
        """Initialize hardware volume to safe level (85%)."""
        try:
            # Set hardware volume to 125 (85% of 147) for optimal quality
            subprocess.run(
                ['amixer', '-c', str(HW_CARD), 'sset', 'PCM', str(HW_VOLUME_MAX)],
                capture_output=True,
                timeout=2
            )
            print(f"[Volume] Hardware initialized to 85% (125/147)")
        except Exception as e:
            print(f"[Volume] Hardware init failed: {e}")
    
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

# Auto-generate a default token so robot is always "claimed"
DEFAULT_TOKEN = "rovy-robot-default-token"

CLAIM_STATE = {
    "claimed": True,  # Auto-claimed by default
    "control_token_hash": hashlib.sha256(DEFAULT_TOKEN.encode()).hexdigest(),
    "pin": None,
    "pin_exp": 0,
}

# Audio state
AUDIO_STATE = {
    "volume": 100,  # 0-100, default 100%
}

# Hardware volume mapping
# Map app volume 0-100% to hardware 0-125 (85% of max 147)
# This prevents overdrive/clipping at max volume
HW_VOLUME_MAX = 125  # 85% of 147
HW_CARD = 3  # USB Audio card

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
            "cloud_stream": WEBSOCKETS_OK,
            "claimed": CLAIM_STATE["claimed"]  # Show robot is claimed
        }
    )


@app.get("/status")
async def get_status():
    """Get robot status with token validation."""
    robot = get_robot()
    
    status_data = {
        "rover_connected": robot.rover is not None,
        "camera_connected": robot.camera is not None,
        "tokenValid": True,  # Always valid since robot is auto-claimed
        "claimed": CLAIM_STATE["claimed"],
        "robotId": "rovy-pi",
        "name": "ROVY"
    }
    
    if robot.rover:
        rover_status = robot.rover.get_status()
        if rover_status:
            voltage = rover_status.get('voltage')
            status_data["battery_voltage"] = voltage
            if voltage:
                status_data["battery_percent"] = robot.rover.voltage_to_percent(voltage)
            status_data["temperature"] = rover_status.get('temperature')
    
    return status_data


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
                    "frame": base64.b64encode(image_bytes).decode('utf-8'),
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
    """Get WiFi connection status with IP address."""
    try:
        # Check if WiFi is connected
        result = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE,STATE", "device", "status"],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_connected = "wifi:connected" in result.stdout
        
        # Get actual SSID
        ssid = "unknown"
        if is_connected:
            try:
                ssid_result = subprocess.run(
                    ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Parse output like "yes:MyWiFiNetwork"
                for line in ssid_result.stdout.split('\n'):
                    if line.startswith('yes:'):
                        ssid = line.split(':', 1)[1]
                        break
            except Exception as e:
                print(f"[WiFi] Could not get SSID: {e}")
        
        # Get IP address
        ip = None
        try:
            ip_result = subprocess.run(
                ["hostname", "-I"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Get first IP address
            ips = ip_result.stdout.strip().split()
            if ips:
                ip = ips[0]
        except Exception as e:
            print(f"[WiFi] Could not get IP: {e}")
        
        return {
            "connected": is_connected,
            "ssid": ssid,
            "ip": ip
        }
    except Exception as e:
        return {"connected": False, "error": str(e), "ssid": "unknown", "ip": None}


@app.get("/mode")
async def get_mode():
    """Get robot operating mode."""
    return {"mode": "autonomous"}


@app.post("/speak")
async def speak_text(request: dict):
    """Speak text using robot's TTS (from cloud AI response)."""
    robot = get_robot()
    
    text = request.get("text", "")
    audio_base64 = request.get("audio", None)
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    print(f"[Speak] {text[:80]}...")
    
    # Run TTS in background thread so it doesn't block API response
    def do_speak():
        try:
            import tempfile
            import os
            
            piper_voice = config.PIPER_VOICE
            
            # Create temp wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                wav_path = f.name
            
            # Generate speech with Piper
            print(f"[Speak] Generating with Piper...")
            proc = subprocess.run(
                ['piper', '--model', piper_voice, '--output_file', wav_path],
                input=text,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if proc.returncode == 0 and os.path.exists(wav_path):
                file_size = os.path.getsize(wav_path)
                print(f"[Speak] Generated {file_size} bytes, playing...")
                
                # Use ffplay - more reliable for full playback
                # -nodisp: no video window, -autoexit: exit when done, -loglevel quiet
                play_proc = subprocess.run(
                    ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', wav_path],
                    capture_output=True,
                    timeout=30
                )
                
                if play_proc.returncode == 0:
                    print(f"[Speak] Played successfully with ffplay")
                else:
                    # Fallback to pw-play
                    print(f"[Speak] Trying pw-play...")
                    subprocess.run(['pw-play', wav_path], capture_output=True, timeout=30)
                    print(f"[Speak] Played with pw-play fallback")
                
                os.unlink(wav_path)
            else:
                error = proc.stderr.decode()[:200] if proc.stderr else "Unknown error"
                print(f"[Speak] Piper failed: {error}")
                
        except Exception as e:
            print(f"[Speak] TTS error: {e}")
            import traceback
            traceback.print_exc()
    
    # Start speaking in background
    threading.Thread(target=do_speak, daemon=True).start()
    
    # Return immediately
    return {"status": "speaking", "method": "piper_async"}


@app.post("/claim/request")
async def claim_request() -> ClaimRequestResponse:
    """Generate a PIN code for claiming the robot (optional, auto-claimed)."""
    # Return a dummy PIN since robot is auto-claimed
    return ClaimRequestResponse(pin="000000")


@app.post("/claim/confirm")
async def claim_confirm(request: ClaimConfirmRequest) -> ClaimConfirmResponse:
    """Confirm PIN and generate control token (auto-approved)."""
    # Auto-approve any PIN since robot is always available
    token = DEFAULT_TOKEN
    
    print("[Claim] Auto-approved claim (no PIN required)")
    return ClaimConfirmResponse(
        control_token=token,
        robot_id="rovy-pi"
    )


@app.post("/claim/release")
async def claim_release():
    """Release the claim (no-op, always available)."""
    print("[Claim] Claim release requested (no-op)")
    return {"released": True}


@app.post("/claim-control")
async def claim_control() -> ClaimControlResponse:
    """Claim control session (auto-approved)."""
    # Auto-approve control
    session_id = secrets.token_urlsafe(16)
    
    print(f"[Claim] Auto-approved control session: {session_id}")
    return ClaimControlResponse(session_id=session_id)


@app.get("/control/volume")
async def get_volume():
    """Get current speaker volume."""
    # Try to read hardware volume and map back to 0-100 scale
    try:
        result = subprocess.run(
            ['amixer', '-c', str(HW_CARD), 'get', 'PCM'],
            capture_output=True,
            text=True,
            timeout=2
        )
        # Parse output to get hardware volume (0-147)
        import re
        match = re.search(r'Playback (\d+) \[(\d+)%\]', result.stdout)
        if match:
            hw_value = int(match.group(1))
            # Map 0-125 to 0-100 (since 125 is our max)
            volume = int((hw_value / HW_VOLUME_MAX) * 100)
            volume = max(0, min(100, volume))  # Clamp to 0-100
            AUDIO_STATE["volume"] = volume
    except Exception as e:
        print(f"[Volume] Could not read hardware volume: {e}")
    
    return {
        "volume": AUDIO_STATE["volume"],
        "min": 0,
        "max": 100
    }


@app.post("/control/volume")
async def set_volume(command: VolumeCommand):
    """Set speaker volume (0-100)."""
    # Clamp volume to valid range
    volume = max(0, min(100, command.volume))
    AUDIO_STATE["volume"] = volume
    
    # Set hardware mixer volume with safe mapping
    # Map 0-100% to 0-125 (prevents overdrive at max volume)
    try:
        hw_volume = int((volume / 100.0) * HW_VOLUME_MAX)
        subprocess.run(
            ['amixer', '-c', str(HW_CARD), 'sset', 'PCM', str(hw_volume)],
            capture_output=True,
            timeout=2
        )
        hw_percent = int((hw_volume / 147.0) * 100)
        print(f"[Volume] Set to {volume}% (hardware: {hw_volume}/147 = {hw_percent}%)")
    except Exception as e:
        print(f"[Volume] Set software volume to {volume}% (hardware control unavailable)")
    
    return {
        "volume": volume,
        "status": "ok"
    }


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
    robot_server.init_volume()
    
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

