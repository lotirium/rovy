"""
Speech Processing - Whisper STT and TTS
Uses local Whisper for speech recognition and espeak/Piper for TTS.
"""
import os
import re
import io
import wave
import tempfile
import subprocess
import logging
from typing import Optional, Any

import numpy as np

logger = logging.getLogger('Speech')

# Try to import Whisper
WHISPER_OK = False
try:
    import whisper
    WHISPER_OK = True
except ImportError:
    logger.warning("Whisper not available. Install: pip install openai-whisper")

# Try to import Piper
PIPER_OK = False
try:
    from piper import PiperVoice
    PIPER_OK = True
except ImportError:
    pass  # Will use espeak fallback


class SpeechProcessor:
    """Speech recognition and synthesis using local models."""
    
    def __init__(self, whisper_model: str = "base", tts_engine: str = "espeak", piper_voices: dict = None):
        self.whisper_model = None
        self.piper_voices = {}  # Dictionary of language -> PiperVoice
        self.piper_voice_paths = piper_voices or {}  # Dictionary of language -> voice path
        self.tts_engine = tts_engine
        
        # Load Whisper
        if WHISPER_OK:
            try:
                logger.info(f"Loading Whisper ({whisper_model})...")
                self.whisper_model = whisper.load_model(whisper_model)
                logger.info("✅ Whisper ready")
            except Exception as e:
                logger.error(f"Whisper load failed: {e}")
        
        # Setup TTS
        if tts_engine == "piper" and PIPER_OK:
            self._init_piper_voices()
        else:
            self._check_espeak()
    
    def _init_piper_voices(self):
        """Initialize Piper TTS voices for multiple languages."""
        if not self.piper_voice_paths:
            logger.info("No Piper voice paths configured, using espeak")
            self._check_espeak()
            return
        
        # Try to load at least one voice (preferably English)
        loaded_count = 0
        for lang, path in self.piper_voice_paths.items():
            if os.path.exists(path):
                try:
                    self.piper_voices[lang] = PiperVoice.load(path)
                    logger.info(f"✅ Piper voice loaded for {lang}: {os.path.basename(path)}")
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load Piper voice for {lang}: {e}")
        
        if loaded_count > 0:
            logger.info(f"✅ Piper ready with {loaded_count} language(s)")
        else:
            logger.info("No Piper voices loaded, using espeak")
            self._check_espeak()
    
    def _load_piper_voice(self, language: str) -> Optional[Any]:
        """Lazy-load a Piper voice for a specific language."""
        # If already loaded, return it
        if language in self.piper_voices:
            return self.piper_voices[language]
        
        # Try to load it
        if language in self.piper_voice_paths:
            path = self.piper_voice_paths[language]
            if os.path.exists(path):
                try:
                    voice = PiperVoice.load(path)
                    self.piper_voices[language] = voice
                    logger.info(f"✅ Loaded Piper voice for {language}: {os.path.basename(path)}")
                    return voice
                except Exception as e:
                    logger.warning(f"Failed to load Piper voice for {language}: {e}")
        
        return None
    
    def _check_espeak(self):
        """Check if espeak is available."""
        try:
            result = subprocess.run(['espeak', '--version'], capture_output=True, timeout=2)
            if result.returncode == 0:
                self.tts_engine = "espeak"
                logger.info("✅ espeak ready")
        except:
            logger.warning("espeak not available")
            self.tts_engine = "none"
    
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe audio to text using Whisper (English only)."""
        if not self.whisper_model:
            logger.error("Whisper not loaded")
            return None
        
        try:
            # Convert bytes to float array
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                from scipy import signal
                # Use scipy's resample for better quality and dtype compatibility
                factor = 16000 / sample_rate
                new_len = int(len(audio) * factor)
                audio = signal.resample(audio, new_len).astype(np.float32)
            
            # Transcribe (English only, better accuracy)
            result = self.whisper_model.transcribe(
                audio,
                language="en",  # Force English
                task="transcribe",
                fp16=False,
                verbose=False,
                temperature=0.0,  # Use greedy decoding for better accuracy
                beam_size=5,  # Use beam search for better results
                best_of=5,  # Sample multiple times and pick best
                condition_on_previous_text=False  # Each segment is independent
            )
            
            text = result["text"].strip()
            if text:
                logger.info(f"Transcribed: '{text}'")
            return text if text else None
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
    
    def synthesize(self, text: str, language: str = "en") -> Optional[bytes]:
        """
        Convert text to speech audio (WAV bytes).
        
        Args:
            text: Text to synthesize
            language: ISO language code (e.g., 'en', 'es', 'fr', 'de', etc.) - default 'en'
        """
        if not text:
            return None
        
        # Preprocess for TTS
        text = self._preprocess(text)
        logger.info(f"Synthesizing ({language}): '{text[:50]}...'")
        
        # Try Piper first if available
        if self.tts_engine == "piper" and PIPER_OK:
            voice = self._load_piper_voice(language)
            if voice:
                return self._synth_piper(text, voice)
            # Fallback to English voice if language not available
            elif language != "en":
                voice = self._load_piper_voice("en")
                if voice:
                    return self._synth_piper(text, voice)
        
        # Fall back to espeak
        return self._synth_espeak(text, language=language)
    
    def _synth_piper(self, text: str, voice: Any) -> Optional[bytes]:
        """Synthesize using Piper with the specified voice."""
        try:
            audio_data = []
            for chunk in voice.synthesize_stream_raw(text):
                audio_data.append(chunk)
            
            if not audio_data:
                return None
            
            raw = b''.join(audio_data)
            
            # Convert to WAV
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                wav.writeframes(raw)
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            return None
    
    def _synth_espeak(self, text: str, speed: int = 150, language: str = "en") -> Optional[bytes]:
        """
        Synthesize using espeak with language support.
        
        Args:
            text: Text to synthesize
            speed: Speech speed
            language: ISO language code (espeak supports 100+ languages)
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            # Language code mapping for espeak
            voice_map = {
                'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 
                'it': 'it', 'pt': 'pt', 'ru': 'ru', 'zh': 'zh',
                'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'hi': 'hi',
                'nl': 'nl', 'pl': 'pl', 'tr': 'tr',
            }
            espeak_voice = voice_map.get(language, 'en')
            
            subprocess.run(
                ['espeak', '-v', espeak_voice, '-w', temp_path, '-s', str(speed), text],
                capture_output=True,
                timeout=30
            )
            
            with open(temp_path, 'rb') as f:
                audio = f.read()
            
            os.unlink(temp_path)
            return audio
            
        except Exception as e:
            logger.error(f"espeak synthesis failed: {e}")
            return None
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for better TTS pronunciation."""
        # Number to words (simple cases)
        def num_to_word(m):
            n = int(m.group(0))
            words = ['zero', 'one', 'two', 'three', 'four', 'five', 
                    'six', 'seven', 'eight', 'nine', 'ten']
            if n < len(words):
                return words[n]
            return m.group(0)
        
        text = re.sub(r'\b(\d)\b', num_to_word, text)
        
        # Abbreviations
        abbrevs = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
        }
        for pat, repl in abbrevs.items():
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

