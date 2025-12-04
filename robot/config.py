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

# Piper TTS voice model paths for different languages
# Download voices from: https://github.com/rhasspy/piper/blob/master/VOICES.md
PIPER_VOICES = {
    "en": "/home/rovy/rovy_client/models/piper/en_US-hfc_male-medium.onnx",
    "es": "/home/rovy/rovy_client/models/piper/es_ES-davefx-medium.onnx",
    "fr": "/home/rovy/rovy_client/models/piper/fr_FR-siwis-medium.onnx",
    "de": "/home/rovy/rovy_client/models/piper/de_DE-thorsten-medium.onnx",
    "it": "/home/rovy/rovy_client/models/piper/it_IT-riccardo-x_low.onnx",
    "pt": "/home/rovy/rovy_client/models/piper/pt_BR-faber-medium.onnx",
    "ru": "/home/rovy/rovy_client/models/piper/ru_RU-dmitri-medium.onnx",
    "zh": "/home/rovy/rovy_client/models/piper/zh_CN-huayan-medium.onnx",
    "vi": "/home/rovy/rovy_client/models/piper/vi_VN-vais1000-medium.onnx",
    "hi": "/home/rovy/rovy_client/models/piper/hi_IN-pratham-medium.onnx",
    "ne": "/home/rovy/rovy_client/models/piper/ne_NP-chitwan-medium.onnx",
    "fa": "/home/rovy/rovy_client/models/piper/fa_IR-amir-medium.onnx",
    # Korean (ko) is not available in Piper TTS
}

# Default voice (backward compatibility)
PIPER_VOICE = PIPER_VOICES.get("en")

# =============================================================================
# Connection
# =============================================================================

RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 0  # 0 = infinite
