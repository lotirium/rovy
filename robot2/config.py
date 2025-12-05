"""
Robot2 Configuration
Runs on Raspberry Pi, provides REST API for Cloud2
"""
import os

# =============================================================================
# Cloud2 Server Connection
# =============================================================================

# Cloud2 server IP (PC running cloud2)
CLOUD2_IP = os.getenv("CLOUD2_IP", "100.121.110.125")

# =============================================================================
# Robot Hardware
# =============================================================================

# Rover serial connection (ESP32)
ROVER_SERIAL_PORT = os.getenv("ROVY_SERIAL_PORT", "/dev/ttyAMA0")
ROVER_BAUDRATE = 115200

# =============================================================================
# Camera
# =============================================================================

CAMERA_INDEX = int(os.getenv("ROVY_CAMERA_INDEX", "1"))
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
JPEG_QUALITY = 80

# =============================================================================
# Audio
# =============================================================================

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# =============================================================================
# Text-to-Speech (Piper)
# =============================================================================

# Piper TTS voice model paths
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
}

# Default voice
PIPER_VOICE = PIPER_VOICES.get("en")

# =============================================================================
# API Server
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000

