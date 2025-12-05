#!/usr/bin/env python3
"""
Robot2 - Raspberry Pi Server for Cloud2
Provides REST API for Cloud2 autonomous android personality.
Main endpoint: /speak (for TTS playback)
"""
import asyncio
import io
import json
import logging
import os
import subprocess
import tempfile
import wave
from typing import Optional, Dict, Any

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")

# Audio playback
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    PLAYBACK_OK = True
except ImportError:
    PLAYBACK_OK = False
    print("WARNING: sounddevice not installed. Audio playback disabled.")

# Piper TTS
try:
    from piper import PiperVoice
    PIPER_OK = True
except ImportError:
    PIPER_OK = False
    print("WARNING: piper-tts not installed. Install with: pip install piper-tts")

# OAK-D Camera
try:
    import depthai as dai
    import cv2
    DEPTHAI_OK = True
except ImportError:
    DEPTHAI_OK = False
    print("WARNING: DepthAI not installed. Camera will not work. Install with: pip install depthai opencv-python")

import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Robot2")

# FastAPI app
if FASTAPI_OK:
    app = FastAPI(title="Robot2 API", version="1.0.0")
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app = None


# =============================================================================
# Pydantic Models
# =============================================================================

class SpeakRequest(BaseModel):
    text: str
    language: str = "en"


# =============================================================================
# TTS Functions
# =============================================================================

# Cache for loaded Piper voices
_piper_voices: Dict[str, Any] = {}

def _load_piper_voice(language: str) -> Optional[Any]:
    """Load a Piper voice for the specified language, with caching."""
    # Check cache first
    if language in _piper_voices:
        return _piper_voices[language]
    
    # Get voice path for language
    voice_path = config.PIPER_VOICES.get(language, config.PIPER_VOICE)
    if not voice_path or not os.path.exists(voice_path):
        if language != "en":
            # Try English as fallback
            voice_path = config.PIPER_VOICE
        if not voice_path or not os.path.exists(voice_path):
            return None
    
    try:
        voice = PiperVoice.load(voice_path)
        _piper_voices[language] = voice
        logger.info(f"✅ Loaded Piper voice for {language}: {os.path.basename(voice_path)}")
        return voice
    except Exception as e:
        logger.error(f"Failed to load Piper voice for {language}: {e}")
        return None

def speak_with_piper(text: str, language: str = "en") -> bool:
    """Synthesize and play speech using Piper TTS."""
    if not text or len(text.strip()) == 0:
        return False
    
    if not PIPER_OK:
        logger.warning("Piper not available, using espeak")
        return speak_with_espeak(text, language)
    
    try:
        # Load voice for language
        voice = _load_piper_voice(language)
        if not voice:
            # Try English as fallback
            if language != "en":
                voice = _load_piper_voice("en")
            if not voice:
                logger.error("No Piper voice found, falling back to espeak")
                return speak_with_espeak(text, language)
        
        logger.info(f"Generating speech with Piper ({language}): {text[:50]}...")
        
        # Synthesize audio using Piper
        audio_data = []
        for chunk in voice.synthesize(text):
            # AudioChunk has audio_int16_bytes attribute
            audio_data.append(chunk.audio_int16_bytes)
        
        if not audio_data:
            logger.error("Piper synthesis produced no audio")
            return speak_with_espeak(text, language)
        
        # Combine all audio chunks
        raw = b''.join(audio_data)
        
        # Convert raw PCM to WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(22050)  # Piper default sample rate
            wav.writeframes(raw)
        
        # Play the generated audio
        if PLAYBACK_OK:
            try:
                # Convert WAV bytes to numpy array for playback
                buf.seek(0)
                data, source_samplerate = sf.read(buf)
                
                # Get device's preferred sample rate
                try:
                    device_info = sd.query_devices(sd.default.device[1], 'output')
                    target_samplerate = int(device_info['default_samplerate'])
                except:
                    target_samplerate = 48000  # Default fallback
                
                # Resample if needed
                if source_samplerate != target_samplerate:
                    from scipy import signal
                    num_samples = int(len(data) * target_samplerate / source_samplerate)
                    data = signal.resample(data, num_samples)
                    samplerate = target_samplerate
                    logger.info(f"Resampled audio from {source_samplerate} Hz to {target_samplerate} Hz")
                else:
                    samplerate = source_samplerate
                
                sd.play(data, samplerate)
                sd.wait()
                logger.info("✅ Speech played successfully")
                return True
            except Exception as e:
                logger.error(f"Audio playback failed: {e}")
                return False
        else:
            logger.warning("Audio playback not available")
            return False
            
    except Exception as e:
        logger.error(f"Piper TTS error: {e}")
        return speak_with_espeak(text, language)


def speak_with_espeak(text: str, language: str = "en") -> bool:
    """Fallback TTS using espeak."""
    try:
        # Language code mapping for espeak
        voice_map = {
            'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 
            'it': 'it', 'pt': 'pt', 'ru': 'ru', 'zh': 'zh',
            'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'hi': 'hi',
        }
        espeak_voice = voice_map.get(language, 'en')
        
        logger.info(f"Using espeak ({language}): {text[:50]}...")
        subprocess.run(
            ['espeak', '-v', espeak_voice, text],
            timeout=30
        )
        return True
    except FileNotFoundError:
        logger.error("espeak not found - no TTS available")
        return False
    except Exception as e:
        logger.error(f"espeak error: {e}")
        return False


# =============================================================================
# Camera Functions
# =============================================================================

# OAK-D camera state
_oakd_device = None
_oakd_queue = None
_camera_initialized = False

def initialize_camera() -> bool:
    """Initialize OAK-D camera."""
    global _oakd_device, _oakd_queue, _camera_initialized
    
    if not DEPTHAI_OK:
        logger.warning("DepthAI not available - camera disabled")
        return False
    
    if _camera_initialized:
        return True
    
    try:
        logger.info("Initializing OAK-D camera...")
        
        # Check for available devices
        available = dai.Device.getAllAvailableDevices()
        if not available:
            logger.error("No OAK-D camera found. Is it connected via USB?")
            return False
        
        logger.info(f"Found {len(available)} OAK-D device(s)")
        
        # Create pipeline
        pipeline = dai.Pipeline()
        
        # Create color camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setIspScale(1, 3)  # Downscale to 640x360 for speed
        cam_rgb.setFps(10)
        
        # Create output
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("preview")
        cam_rgb.preview.link(xout_rgb.input)
        
        # Connect to device
        _oakd_device = dai.Device(pipeline)
        _oakd_queue = _oakd_device.getOutputQueue(name="preview", maxSize=4, blocking=False)
        
        # Wait for first frame
        import time
        time.sleep(0.5)
        frame_data = _oakd_queue.tryGet()
        if frame_data is not None:
            frame = frame_data.getCvFrame()
            if frame is not None:
                _camera_initialized = True
                logger.info("✅ OAK-D camera initialized successfully")
                return True
        
        logger.warning("Camera initialized but no frames received yet")
        _camera_initialized = True  # Still mark as initialized
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize OAK-D camera: {e}", exc_info=True)
        return False

def capture_frame() -> Optional[bytes]:
    """Capture a frame from OAK-D camera and return as JPEG bytes."""
    global _oakd_device, _oakd_queue, _camera_initialized
    
    if not _camera_initialized:
        if not initialize_camera():
            return None
    
    if _oakd_queue is None:
        return None
    
    try:
        frame_data = _oakd_queue.tryGet()
        if frame_data is not None:
            frame = frame_data.getCvFrame()
            if frame is not None and frame.size > 0:
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return buffer.tobytes()
    except Exception as e:
        logger.warning(f"Frame capture error: {e}")
    
    return None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "ok",
        "service": "Robot2",
        "description": "Raspberry Pi server for Cloud2 autonomous android",
        "endpoints": {
            "/speak": "POST - Text to speech",
            "/frame": "GET - Camera frame (JPEG)",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Robot2",
        "tts_available": PLAYBACK_OK,
        "camera_available": _camera_initialized
    }

@app.get("/frame")
async def get_frame():
    """
    Get a single camera frame from OAK-D as JPEG.
    
    Returns JPEG image bytes for Cloud2 vision processing.
    """
    frame_bytes = capture_frame()
    if frame_bytes is None:
        raise HTTPException(status_code=503, detail="Camera not available or failed to capture frame")
    
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.post("/speak")
async def speak(request: SpeakRequest):
    """
    Text-to-speech endpoint for Cloud2.
    
    Receives text from Cloud2 and plays it through Pi speakers.
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="No text provided")
    
    logger.info(f"🔊 Speaking ({request.language}): {request.text[:60]}...")
    
    # Use Piper TTS (with espeak fallback)
    success = speak_with_piper(request.text, request.language)
    
    if success:
        return {
            "status": "ok",
            "message": "Speech played",
            "text": request.text[:100],
            "language": request.language
        }
    else:
        raise HTTPException(status_code=503, detail="TTS not available")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    if not FASTAPI_OK:
        logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
        return 1
    
    logger.info("=" * 60)
    logger.info("Robot2 - Raspberry Pi Server for Cloud2")
    logger.info("=" * 60)
    logger.info(f"API Server: http://{config.API_HOST}:{config.API_PORT}")
    logger.info(f"Cloud2 IP: {config.CLOUD2_IP}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  POST /speak - Text to speech (for Cloud2)")
    logger.info("  GET  /frame - Camera frame (JPEG)")
    logger.info("  GET  /health - Health check")
    logger.info("")
    
    # Initialize camera on startup
    if DEPTHAI_OK:
        initialize_camera()
    logger.info("Starting server...")
    logger.info("=" * 60)
    
    try:
        uvicorn.run(
            app,
            host=config.API_HOST,
            port=config.API_PORT,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

