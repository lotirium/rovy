"""
Cloud 2 Configuration
Autonomous Personality Service Settings
"""
import os

# =============================================================================
# Pi Connection (same as main cloud)
# =============================================================================

# Raspberry Pi IP address (via Tailscale or local network)
PI_IP = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")
PI_SPEAK_URL = f"http://{PI_IP}:8000/speak"

# =============================================================================
# OpenAI Configuration
# =============================================================================

# OpenAI API Key (REQUIRED)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model to use (gpt-4o recommended for vision, gpt-4-turbo for text-only)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# =============================================================================
# Personality Configuration
# =============================================================================

# Personality name (your android's name)
# Default: "Android" - you can set to any name you want
PERSONALITY_NAME = os.getenv("PERSONALITY_NAME", "Android")

# Observation interval (seconds between observations)
OBSERVATION_INTERVAL = float(os.getenv("PERSONALITY_OBS_INTERVAL", "8.0"))

# Speech cooldown (minimum seconds between speaking)
SPEECH_COOLDOWN = float(os.getenv("PERSONALITY_SPEECH_COOLDOWN", "4.0"))

# Enable vision (OAK D camera)
VISION_ENABLED = os.getenv("PERSONALITY_VISION", "true").lower() == "true"

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

