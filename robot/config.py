"""
Rovy Robot Client Configuration
Runs on Raspberry Pi, connects to cloud server via Tailscale
"""
import os

# =============================================================================
# Cloud Server Connection (via Tailscale)
# =============================================================================

# Your PC's Tailscale IP
PC_SERVER_IP = os.getenv("ROVY_PC_IP", "100.121.110.125")
WS_PORT = 8765

# WebSocket URL
SERVER_URL = f"ws://{PC_SERVER_IP}:{WS_PORT}"

# =============================================================================
# Robot Hardware
# =============================================================================

# Rover serial connection (ESP32)
# Pi5 uses /dev/ttyAMA0 for GPIO UART, older Pis may use /dev/ttyS0 or /dev/ttyACM0
ROVER_SERIAL_PORT = os.getenv("ROVY_SERIAL_PORT", "/dev/ttyAMA0")
ROVER_BAUDRATE = 115200

# =============================================================================
# Camera
# =============================================================================

CAMERA_INDEX = int(os.getenv("ROVY_CAMERA_INDEX", "1"))  # USB Camera is at /dev/video1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
JPEG_QUALITY = 80

# =============================================================================
# Audio (ReSpeaker)
# =============================================================================

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_BUFFER_SECONDS = 2.0

# =============================================================================
# Text-to-Speech (Piper)
# =============================================================================

# Piper TTS voice model path
PIPER_VOICE = os.getenv("ROVY_PIPER_VOICE", "/home/rovy/rovy_client/models/piper/en_US-hfc_male-medium.onnx")

# =============================================================================
# Connection
# =============================================================================

RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 0  # 0 = infinite
