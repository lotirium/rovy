"""
Shared Configuration for Rovy Cloud System
Uses LOCAL models - no cloud APIs needed!
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ServerConfig:
    """Cloud server configuration - LOCAL MODELS"""
    host: str = "0.0.0.0"
    port: int = 8765
    
    # LOCAL LLM paths (llama.cpp GGUF format)
    # Text model - Gemma, Llama, Mistral, etc.
    local_llm_path: Optional[str] = field(default_factory=lambda: os.getenv(
        "ROVY_TEXT_MODEL",
        None  # Will auto-detect in assistant.py
    ))
    
    # Vision model - LLaVA, Phi-3-Vision, etc.
    local_vision_model_path: Optional[str] = field(default_factory=lambda: os.getenv(
        "ROVY_VISION_MODEL",
        None  # Will auto-detect
    ))
    
    # Vision projector for LLaVA
    local_vision_mmproj_path: Optional[str] = field(default_factory=lambda: os.getenv(
        "ROVY_VISION_MMPROJ",
        None  # Will auto-detect
    ))
    
    # Speech settings (LOCAL Whisper)
    whisper_model: str = "base"  # tiny, base, small, medium, large
    
    # TTS settings (LOCAL Piper/espeak)
    tts_engine: str = "piper"  # piper, espeak
    piper_voice_path: Optional[str] = None  # Auto-detect
    
    # Face recognition
    known_faces_dir: str = "known_faces"
    face_recognition_tolerance: float = 0.6  # Lower = stricter
    
    # Performance
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    n_ctx: int = 2048  # Context window
    max_workers: int = 4
    inference_timeout: int = 60


@dataclass
class AssistantConfig:
    """AI Assistant behavior configuration"""
    name: str = "Rovy"
    wake_words: List[str] = field(default_factory=lambda: ["hey rovy", "rovy", "hey robot"])
    
    # Response settings
    max_tokens: int = 150
    temperature: float = 0.7
    
    # System prompt for the assistant
    system_prompt: str = """You are Rovy, a friendly and intelligent robot assistant.
You have a camera to see the world, can move around on wheels, and speak to users.

Keep responses concise and natural - under 50 words for conversation.
Be helpful, curious, and occasionally playful."""


# Default instances
server_config = ServerConfig()
assistant_config = AssistantConfig()
