"""
Speech Processing Module - Using Local Models
Handles speech recognition (Whisper) and TTS (Piper/espeak)
Ported from original rovy/smart_assistant.py
"""
import os
import io
import re
import wave
import logging
import tempfile
import subprocess
from typing import Optional
import numpy as np

logger = logging.getLogger('Speech')

# Try to import Whisper for STT
WHISPER_AVAILABLE = False
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.info("Whisper not available. Install with: pip install openai-whisper")

# Try to import Piper for TTS
PIPER_AVAILABLE = False
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    logger.info("Piper TTS not available. Install with: pip install piper-tts")


class SpeechProcessor:
    """
    Handles speech-to-text and text-to-speech using local models.
    - Whisper for speech recognition
    - Piper for natural TTS
    - espeak as fallback TTS
    
    Ported from rovy/smart_assistant.py
    """
    
    def __init__(self, 
                 whisper_model: str = "base",
                 piper_voice_path: str = None,
                 tts_engine: str = "piper"):
        """
        Initialize speech processor with local models.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            piper_voice_path: Path to Piper voice model (.onnx file)
            tts_engine: TTS engine to use (piper, espeak)
        """
        self.whisper_model = None
        self.piper_voice = None
        self.tts_engine = tts_engine
        
        # Initialize Whisper for STT
        if WHISPER_AVAILABLE:
            try:
                logger.info(f"Loading Whisper model ({whisper_model})...")
                self.whisper_model = whisper.load_model(whisper_model)
                logger.info("✅ Whisper model loaded")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
        else:
            logger.warning("Whisper not available - speech recognition disabled")
        
        # Initialize Piper TTS
        if PIPER_AVAILABLE and tts_engine == "piper":
            self._init_piper(piper_voice_path)
        
        # Check for espeak fallback
        if not self.piper_voice:
            self._check_espeak()
    
    def _init_piper(self, voice_path: str = None):
        """Initialize Piper TTS with a voice model"""
        # Search for voice models
        voice_locations = [
            voice_path,
            os.path.expanduser("~/.local/share/piper-voices/en_US-hfc_male-medium.onnx"),
            os.path.expanduser("~/.local/share/piper-voices/en_US-danny-low.onnx"),
            os.path.expanduser("~/.local/share/piper-voices/en_US-ryan-high.onnx"),
            os.path.expanduser("~/.local/share/piper-voices/en_US-libritts_r-medium.onnx"),
            os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx"),
            "./voices/en_US-lessac-medium.onnx",
        ]
        
        for path in voice_locations:
            if path and os.path.exists(path):
                try:
                    logger.info(f"Loading Piper voice: {os.path.basename(path)}")
                    self.piper_voice = PiperVoice.load(path, use_cuda=False)
                    logger.info("✅ Piper TTS ready")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load Piper voice: {e}")
        
        logger.warning("No Piper voice found - will use espeak fallback")
    
    def _check_espeak(self):
        """Check if espeak is available"""
        try:
            result = subprocess.run(['espeak', '--version'], capture_output=True, timeout=2)
            if result.returncode == 0:
                logger.info("espeak available as TTS fallback")
                self.tts_engine = "espeak"
        except Exception:
            logger.warning("espeak not available")
            self.tts_engine = "print"
    
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_bytes: Raw audio data (int16 format)
            sample_rate: Audio sample rate
            
        Returns:
            str: Transcribed text or None
        """
        if not self.whisper_model:
            logger.error("Whisper model not loaded")
            return None
        
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                factor = 16000 / sample_rate
                new_length = int(len(audio_data) * factor)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio_data,
                language="en",
                task="transcribe",
                fp16=False,
                verbose=False
            )
            
            text = result["text"].strip()
            logger.info(f"📝 Transcribed: '{text}'")
            return text if text else None
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return None
    
    def synthesize(self, text: str) -> Optional[bytes]:
        """
        Convert text to speech audio using local TTS.
        
        Args:
            text: Text to synthesize
            
        Returns:
            bytes: WAV audio data or None
        """
        if not text:
            return None
        
        # Preprocess text for better pronunciation
        processed_text = self._preprocess_text(text)
        
        logger.info(f"🔊 Synthesizing: '{processed_text[:50]}...'")
        
        if self.piper_voice and PIPER_AVAILABLE:
            return self._synthesize_piper(processed_text)
        elif self.tts_engine == "espeak":
            return self._synthesize_espeak(processed_text)
        else:
            logger.warning("No TTS engine available")
            return None
    
    def _synthesize_piper(self, text: str) -> Optional[bytes]:
        """Synthesize using Piper TTS (high-quality, natural voice)"""
        try:
            from piper.config import SynthesisConfig
            
            # Configure for natural sounding speech
            config = SynthesisConfig(
                noise_scale=0.667,
                length_scale=0.9,  # Slightly faster
                noise_w_scale=0.8,
            )
            
            # Generate audio
            audio_data = []
            for audio_bytes in self.piper_voice.synthesize_stream_raw(text):
                audio_data.append(audio_bytes)
            
            if not audio_data:
                return None
            
            raw_audio = b''.join(audio_data)
            
            # Convert to WAV
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(22050)  # Piper default
                wav_file.writeframes(raw_audio)
            
            return wav_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Piper synthesis error: {e}")
            return None
    
    def _synthesize_espeak(self, text: str, speed: int = 150) -> Optional[bytes]:
        """Synthesize using espeak (fast, robotic voice)"""
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            subprocess.run(
                ['espeak', '-w', temp_path, '-s', str(speed), text],
                capture_output=True,
                timeout=30
            )
            
            # Read the file
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"espeak synthesis error: {e}")
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for more natural TTS pronunciation.
        Ported from rovy/smart_assistant.py
        """
        # Convert numbers to words for better pronunciation
        def number_to_words(num_str):
            try:
                num = int(num_str)
                if num < 20:
                    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
                            'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
                            'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
                    return words[num] if num < len(words) else num_str
                elif num < 100:
                    tens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
                    ones = num % 10
                    ten = num // 10
                    if ones == 0:
                        return tens[ten - 2]
                    return f"{tens[ten - 2]} {number_to_words(str(ones))}"
                else:
                    return num_str
            except:
                return num_str
        
        # Replace standalone numbers (1-99) with words
        text = re.sub(r'\b(\d{1,2})\b', lambda m: number_to_words(m.group(1)), text)
        
        # Expand common abbreviations
        abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\bProf\.': 'Professor',
            r'\bvs\.': 'versus',
            r'\betc\.': 'etcetera',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_wake_word(self, audio_bytes: bytes, sample_rate: int, 
                         wake_words: list = None) -> Optional[str]:
        """
        Detect wake word in audio.
        
        Args:
            audio_bytes: Raw audio data
            sample_rate: Audio sample rate
            wake_words: List of wake words to detect
            
        Returns:
            str: Detected wake word or None
        """
        if wake_words is None:
            wake_words = ["hey rovy", "rovy", "hey robot"]
        
        # Transcribe the audio
        text = self.transcribe(audio_bytes, sample_rate)
        
        if text:
            text_lower = text.lower()
            for wake_word in wake_words:
                if wake_word in text_lower:
                    logger.info(f"👋 Wake word detected: '{wake_word}'")
                    return wake_word
        
        return None


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = SpeechProcessor(whisper_model="tiny")
    
    # Test TTS
    print("\n--- TTS Test ---")
    audio = processor.synthesize("Hello! I am Rovy, your robot assistant.")
    if audio:
        print(f"Generated {len(audio)} bytes of audio")
        with open("test_speech.wav", "wb") as f:
            f.write(audio)
        print("Saved to test_speech.wav")
    else:
        print("TTS not available")
