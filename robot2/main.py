#!/usr/bin/env python3
"""
Robot2 - Raspberry Pi Server for Cloud2
Provides REST API for Cloud2 autonomous android personality.
Main endpoint: /speak (for TTS playback)
"""
import asyncio
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
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
    PLAYBACK_OK = True
except ImportError:
    PLAYBACK_OK = False
    print("WARNING: sounddevice not installed. Audio playback disabled.")

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

def speak_with_piper(text: str, language: str = "en") -> bool:
    """Synthesize and play speech using Piper TTS."""
    if not text or len(text.strip()) == 0:
        return False
    
    try:
        # Get voice path for language
        voice_path = config.PIPER_VOICES.get(language, config.PIPER_VOICE)
        if not voice_path or not os.path.exists(voice_path):
            logger.warning(f"Piper voice not found for {language}, using English")
            voice_path = config.PIPER_VOICE
        
        if not voice_path or not os.path.exists(voice_path):
            logger.error("No Piper voice found, falling back to espeak")
            return speak_with_espeak(text, language)
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
        
        # Generate speech with Piper
        logger.info(f"Generating speech with Piper ({language}): {text[:50]}...")
        proc = subprocess.run(
            ['piper', '--model', voice_path, '--output_file', wav_path],
            input=text,
            text=True,
            capture_output=True,
            timeout=30
        )
        
        if proc.returncode == 0 and os.path.exists(wav_path):
            # Play the generated audio
            if PLAYBACK_OK:
                try:
                    data, samplerate = sf.read(wav_path)
                    sd.play(data, samplerate)
                    sd.wait()
                    logger.info("✅ Speech played successfully")
                    os.unlink(wav_path)
                    return True
                except Exception as e:
                    logger.error(f"Audio playback failed: {e}")
                    os.unlink(wav_path)
                    return False
            else:
                logger.warning("Audio playback not available")
                os.unlink(wav_path)
                return False
        else:
            logger.warning(f"Piper failed: {proc.stderr.decode()}")
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            return speak_with_espeak(text, language)
            
    except FileNotFoundError:
        logger.warning("Piper not found, using espeak")
        return speak_with_espeak(text, language)
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
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Robot2",
        "tts_available": PLAYBACK_OK
    }


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
    logger.info("  GET  /health - Health check")
    logger.info("")
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

