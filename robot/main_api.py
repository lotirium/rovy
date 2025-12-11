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
    from wake_word_detector_cloud import CloudWakeWordDetector
    WAKE_WORD_OK = True
except ImportError:
    WAKE_WORD_OK = False
    print("WARNING: Wake word detector not available. Install with: bash install_wake_word.sh")

try:
    from rover import Rover
    ROVER_OK = True
except Exception as e:
    ROVER_OK = False
    print(f"WARNING: Rover not available: {e}")

try:
    import depthai as dai
    DEPTHAI_OK = True
except ImportError:
    DEPTHAI_OK = False
    print("WARNING: DepthAI not installed. OAK-D camera disabled.")


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

class NavigationCommand(BaseModel):
    action: str  # start_explore, stop, goto
    duration: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None


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
        self.current_gesture = 'none'  # Current detected gesture
        self.gesture_confidence = 0.0
        self.rover = None
        self.camera = None
        self.camera_type = None  # 'oakd' or 'usb'
        self.usb_camera = None  # USB camera instance (always separate from OAK-D)
        self.oakd_device = None
        self.oakd_queue = None
        self.audio_stream = None
        
        # State
        self.is_listening = False
        self.is_speaking = False  # Flag to pause wake word detection during TTS
        self.audio_buffer = []
        self.last_image = None
        self.last_image_time = 0
        self.oakd_error_count = 0
        self.oakd_last_reinit = 0
        
        # Wake word detection
        self.wake_word_model = None
        self.wake_word_enabled = False
        self.is_recording = False
        self.voice_ws = None  # WebSocket connection to cloud /voice endpoint
        self.audio_sample_rate = config.SAMPLE_RATE  # Default, will be updated in init_audio
        
        # Navigation state (for USB bandwidth management)
        self.is_navigating = False  # True when OAK-D navigation is active
        
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
        """Initialize camera - USB for streaming, OAK-D lazy-loaded only when needed."""
        # Skip OAK-D initialization at startup - it will be lazy-loaded when needed
        # This prevents resource waste and issues when OAK-D is not required
        print("[Camera] OAK-D will be initialized on-demand for vision commands")
        oakd_initialized = False
        
        # Always try to initialize USB camera (for /snapshot endpoint, even when OAK-D is present)
        usb_initialized = False
        if CAMERA_OK:
            print("[Camera] Trying USB camera (for /snapshot endpoint)...")
            # Try indices in order: 1 (USB Camera), 0, 2
            for camera_index in [1, 0, 2]:
                try:
                    print(f"[Camera] Trying /dev/video{camera_index} for USB camera...")
                    usb_cam = cv2.VideoCapture(camera_index)
                    usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                    usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                    usb_cam.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                    
                    ret, frame = usb_cam.read()
                    if ret and frame is not None:
                        self.usb_camera = usb_cam
                        print(f"[Camera] USB camera ready on /dev/video{camera_index}")
                        usb_initialized = True
                        break
                    else:
                        usb_cam.release()
                except Exception as e:
                    print(f"[Camera] USB camera init failed on /dev/video{camera_index}: {e}")
                    try:
                        usb_cam.release()
                    except:
                        pass
        
        # If OAK-D is not available, use USB camera as primary
        if self.camera_type != 'oakd':
            if usb_initialized:
                self.camera = self.usb_camera
                self.camera_type = 'usb'
                return True
            else:
                print("[Camera] No working camera found")
                return False
        else:
            # OAK-D is primary, USB is for /snapshot endpoint
            return True
    
    def init_audio(self):
        """Initialize audio input (microphone)."""
        if not AUDIO_OK:
            return False
        
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Find microphone device (USB Headphone Set or ReSpeaker)
            device_index = None
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                name = info.get('name', '').lower()
                max_input_channels = info.get('maxInputChannels', 0)
                
                # Prefer USB Headphone Set, fallback to ReSpeaker
                if device_index is None and max_input_channels > 0:
                    if 'headphone' in name or 'set' in name:
                        device_index = i
                        print(f"[Audio] Found USB Headphone Set: {info['name']}")
                    elif 'respeaker' in name or 'seeed' in name:
                        device_index = i
                        print(f"[Audio] Found ReSpeaker: {info['name']}")
            
            if device_index is None:
                # Fallback: use default input device
                try:
                    default_info = self.pyaudio.get_default_input_device_info()
                    device_index = default_info.get('index')
                    print(f"[Audio] Using default input device: {default_info.get('name')}")
                except:
                    print("[Audio] No microphone found")
                    return False
            
            self.audio_device_index = device_index
            
            # Get device's default sample rate
            device_info = self.pyaudio.get_device_info_by_index(device_index)
            device_sample_rate = int(device_info.get('defaultSampleRate', 48000))
            self.audio_sample_rate = device_sample_rate
            
            # If device doesn't support 16kHz, we'll record at device rate and resample
            if device_sample_rate != config.SAMPLE_RATE:
                print(f"[Audio] Device sample rate: {device_sample_rate}Hz (will resample to {config.SAMPLE_RATE}Hz)")
            else:
                print(f"[Audio] Using device sample rate: {device_sample_rate}Hz")
            
            print(f"[Audio] Ready (device index: {device_index})")
            return True
        except Exception as e:
            print(f"[Audio] Init failed: {e}")
            return False
    
    def init_volume(self):
        """Initialize hardware volume to safe level (85%)."""
        global HW_CARD
        # Re-detect card in case it changed
        if HW_CARD is None:
            HW_CARD = detect_audio_card()
        
        if HW_CARD is None:
            print("[Volume] ⚠️  No audio card detected, skipping hardware init")
            return
        
        try:
            # Set hardware volume to 125 (85% of 147) to prevent clipping
            result = subprocess.run(
                ['amixer', '-c', str(HW_CARD), 'sset', 'PCM', str(HW_VOLUME_MAX)],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                print(f"[Volume] Hardware initialized to 85% (125/147) on card {HW_CARD}")
            else:
                print(f"[Volume] Hardware init failed: {result.stderr}")
        except Exception as e:
            print(f"[Volume] Hardware init failed: {e}")
    
    def speak_acknowledgment(self, text="Yes?"):
        """Quick non-blocking speech for acknowledgments using Piper."""
        def do_speak():
            try:
                import tempfile
                import os
                
                self.is_speaking = True  # Pause wake word detection
                print(f"[WakeWord] Playing acknowledgment: '{text}'")
                piper_voice = config.PIPER_VOICES.get("en")
                if not os.path.exists(piper_voice):
                    print(f"[WakeWord] Piper voice not found: {piper_voice}")
                    self.is_speaking = False
                    return
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    wav_path = f.name
                
                # Generate with Piper
                proc = subprocess.run(
                    ['piper', '--model', piper_voice, '--output_file', wav_path],
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=5
                )
                
                if proc.returncode == 0 and os.path.exists(wav_path):
                    print(f"[WakeWord] Generated audio, playing on {config.SPEAKER_DEVICE}")
                    # Play with aplay
                    play_result = subprocess.run(
                        ['aplay', '-D', config.SPEAKER_DEVICE, wav_path],
                        capture_output=True,
                        timeout=5
                    )
                    if play_result.returncode == 0:
                        print(f"[WakeWord] ✅ Acknowledgment played")
                    else:
                        print(f"[WakeWord] aplay error: {play_result.stderr.decode()[:100]}")
                    os.unlink(wav_path)
                else:
                    print(f"[WakeWord] Piper generation failed: {proc.stderr.decode()[:100]}")
            except Exception as e:
                print(f"[WakeWord] Acknowledgment error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Wait 0.5s for audio to clear from mic before resuming detection
                import time
                time.sleep(0.5)
                self.is_speaking = False  # Resume wake word detection
        
        # Run in background thread
        threading.Thread(target=do_speak, daemon=True).start()
    
    def init_wake_word(self):
        """Initialize wake word detection with Silero VAD + Whisper."""
        if not WAKE_WORD_OK:
            print("[WakeWord] Wake word detection not available")
            return False
        
        try:
            print("[WakeWord] Loading Cloud-based Wake Word Detector (Silero VAD + Cloud Whisper)...")
            device_rate = self.audio_sample_rate if hasattr(self, 'audio_sample_rate') else config.VAD_SAMPLE_RATE
            device_idx = self.audio_device_index if hasattr(self, 'audio_device_index') else None
            print(f"[WakeWord] Config: device_rate={device_rate}, device_index={device_idx}, vad_rate={config.VAD_SAMPLE_RATE}")
            self.wake_word_detector = CloudWakeWordDetector(
                wake_words=config.WAKE_WORDS,
                cloud_url=f"ws://{config.PC_SERVER_IP}:{config.API_PORT}/voice",
                sample_rate=config.VAD_SAMPLE_RATE,
                device_sample_rate=device_rate,
                device_index=device_idx,
                vad_threshold=config.VAD_THRESHOLD,
                min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
                min_silence_duration=config.VAD_MIN_SILENCE_DURATION,
            )
            self.wake_word_enabled = True
            print(f"[WakeWord] Ready! Listening for: {config.WAKE_WORDS}")
            return True
        except Exception as e:
            print(f"[WakeWord] Init failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def connect_voice_websocket(self):
        """Connect to cloud /voice WebSocket endpoint."""
        if not WEBSOCKETS_OK:
            return False
        
        try:
            # Get cloud HTTP URL and convert to WebSocket
            cloud_http_url = config.SERVER_URL.replace('ws://', 'http://').replace('wss://', 'https://')
            cloud_http_url = cloud_http_url.replace(':8765', ':8000')  # FastAPI runs on 8000
            voice_ws_url = cloud_http_url.replace('http://', 'ws://').replace('https://', 'wss://')
            voice_ws_url = f"{voice_ws_url}/voice"
            
            print(f"[Voice] Connecting to {voice_ws_url}...")
            self.voice_ws = await websockets.connect(voice_ws_url)
            print("[Voice] Connected to cloud voice endpoint")
            return True
        except Exception as e:
            print(f"[Voice] Connection failed: {e}")
            self.voice_ws = None
            return False
    
    def calculate_audio_energy(self, audio_data):
        """Calculate audio energy for VAD (Voice Activity Detection)."""
        if not audio_data:
            return 0.0
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return np.sqrt(np.mean(audio_array**2))
    
    async def record_and_stream_audio(self, duration=5.0, silence_threshold=500, silence_duration=1.5):
        """Record audio and stream to cloud /voice endpoint."""
        if not AUDIO_OK or not self.pyaudio or self.audio_device_index is None:
            print("[Voice] Audio not available")
            return
        
        if not self.voice_ws:
            if not await self.connect_voice_websocket():
                print("[Voice] Could not connect to cloud")
                return
        
        print("[Voice] 🎤 Recording...")
        self.is_recording = True
        
        try:
            # Try to open stream
            try:
                stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=config.CHANNELS,
                    rate=self.audio_sample_rate,  # Use device's native sample rate
                    input=True,
                    input_device_index=self.audio_device_index,
                    frames_per_buffer=config.CHUNK_SIZE
                )
            except Exception as stream_error:
                print(f"[Voice] Failed to open audio stream: {stream_error}")
                return
            
            audio_chunks = []
            chunk_count = 0
            last_silence_time = None
            start_time = time.time()
            
            while self.is_recording and (time.time() - start_time) < duration:
                try:
                    audio_data = stream.read(config.CHUNK_SIZE, exception_on_overflow=False)
                    energy = self.calculate_audio_energy(audio_data)
                    
                    # Check for silence
                    if energy < silence_threshold:
                        if last_silence_time is None:
                            last_silence_time = time.time()
                        elif (time.time() - last_silence_time) > silence_duration:
                            # Silence detected, stop recording
                            print("[Voice] Silence detected, stopping recording")
                            break
                    else:
                        # Audio detected, reset silence timer
                        last_silence_time = None
                    
                    # Resample if needed (device rate -> 16kHz for cloud)
                    if self.audio_sample_rate != config.SAMPLE_RATE:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        # Resample using scipy if available
                        try:
                            from scipy import signal
                            num_samples = int(len(audio_array) * config.SAMPLE_RATE / self.audio_sample_rate)
                            audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
                            audio_data = audio_array.tobytes()
                        except ImportError:
                            # Fallback: simple decimation (not ideal but works)
                            decimation_factor = int(self.audio_sample_rate / config.SAMPLE_RATE)
                            audio_array = audio_array[::decimation_factor]
                            audio_data = audio_array.tobytes()
                    
                    # Encode chunk as base64
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    audio_chunks.append(audio_b64)
                    chunk_count += 1
                    
                    # Send chunk to cloud
                    try:
                        await self.voice_ws.send(json.dumps({
                            "type": "audio_chunk",
                            "encoding": "base64",
                            "data": audio_b64
                        }))
                    except Exception as e:
                        print(f"[Voice] Error sending chunk: {e}")
                        break
                    
                except Exception as e:
                    print(f"[Voice] Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
            # Send audio_end signal
            if chunk_count > 0:
                try:
                    await self.voice_ws.send(json.dumps({
                        "type": "audio_end",
                        "encoding": "base64",
                        "sampleRate": config.SAMPLE_RATE
                    }))
                    print(f"[Voice] ✅ Sent {chunk_count} audio chunks to cloud")
                except Exception as e:
                    print(f"[Voice] Error sending audio_end: {e}")
            
        except Exception as e:
            print(f"[Voice] Recording failed: {e}")
        finally:
            self.is_recording = False
    
    async def listen_for_wake_word(self):
        """Continuously listen for wake word 'Rovy'."""
        if not AUDIO_OK or not self.pyaudio or self.audio_device_index is None:
            return
        
        if not self.wake_word_enabled:
            return
        
        print("[WakeWord] 👂 Listening for 'Rovy'...")
        
        try:
            stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=config.CHANNELS,
                rate=self.audio_sample_rate,  # Use device's native sample rate
                input=True,
                output=False,  # Explicitly input-only
                input_device_index=self.audio_device_index,
                frames_per_buffer=config.CHUNK_SIZE
            )
            
            # Buffer for accumulating audio (for text-based detection)
            audio_buffer = []
            buffer_duration = 3.0  # seconds
            buffer_size = int(config.SAMPLE_RATE / config.CHUNK_SIZE * buffer_duration)
            
            while self.running and not self.is_recording:
                try:
                    audio_data = stream.read(config.CHUNK_SIZE, exception_on_overflow=False)
                    energy = self.calculate_audio_energy(audio_data)
                    
                    # Simple VAD: if energy is above threshold, accumulate audio
                    vad_threshold = 300  # Adjust based on your microphone
                    if energy > vad_threshold:
                        audio_buffer.append(audio_data)
                        # Keep buffer size limited
                        if len(audio_buffer) > buffer_size:
                            audio_buffer.pop(0)
                    else:
                        # Check buffer for wake word if we have enough audio
                        if len(audio_buffer) > int(config.SAMPLE_RATE / config.CHUNK_SIZE * 0.5):  # At least 0.5s
                            # Try to detect wake word using Whisper (via cloud)
                            # For now, use a simpler approach: send to cloud for transcription
                            # and check for "rovy" in the text
                            
                            # Combine buffer audio
                            combined_audio = b''.join(audio_buffer)
                            audio_buffer = []  # Clear buffer
                            
                            # Quick check: send to cloud for transcription
                            # We'll use a lightweight approach: send short audio clip
                            # to cloud /voice endpoint for wake word detection
                            
                            # For efficiency, we'll use a simpler method:
                            # Check if we should trigger based on pattern
                            # For now, let's use a hybrid: if energy spikes, check with cloud
                            
                            # Actually, let's use a simpler approach:
                            # Periodically send audio to cloud for wake word detection
                            # But to avoid too many requests, we'll only do this when energy is high
                            
                            # For now, implement a simple pattern: if we detect speech-like
                            # patterns, send to cloud for transcription
                            pass
                    
                    await asyncio.sleep(0.01)  # Small delay to avoid busy waiting
                    
                except Exception as e:
                    print(f"[WakeWord] Error: {e}")
                    await asyncio.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"[WakeWord] Listening failed: {e}")
    
    async def wake_word_detection_loop(self):
        """Main loop for wake word detection using Cloud-based detector."""
        if not self.wake_word_enabled or not hasattr(self, 'wake_word_detector'):
            print("[WakeWord] Wake word detector not available")
            return
        
        # Connect to voice WebSocket for sending queries after wake word
        await self.connect_voice_websocket()
        
        print("[WakeWord] 👂 Starting cloud-based wake word detection (Silero VAD + Cloud Whisper)...")
        
        while self.running and not self.is_recording:
            try:
                # Skip detection if robot is speaking (prevent hearing own voice)
                if self.is_speaking:
                    # Don't spam logs, just wait quietly
                    await asyncio.sleep(0.1)
                    continue
                
                # Run wake word detection using async cloud detector
                detected = await self.wake_word_detector.listen_for_wake_word_async(timeout=10)
                
                if detected and not self.is_recording:
                    print("[WakeWord] ✅ Wake word detected!")
                    
                    # Play acknowledgment via Piper (non-blocking)
                    try:
                        await asyncio.to_thread(self.speak_acknowledgment, "Yes?")
                        await asyncio.sleep(0.5)  # Brief pause for acknowledgment to play
                    except Exception as e:
                        print(f"[WakeWord] Could not play acknowledgment: {e}")
                    
                    # Record full query using pyaudio
                    print("[WakeWord] 🎤 Recording your question...")
                    try:
                        # Record audio for query
                        frames = []
                        stream = self.pyaudio.open(
                            format=pyaudio.paInt16,
                            channels=config.CHANNELS,
                            rate=self.audio_sample_rate,
                            input=True,
                            input_device_index=self.audio_device_index,
                            frames_per_buffer=config.CHUNK_SIZE
                        )
                        
                        num_chunks = int(self.audio_sample_rate / config.CHUNK_SIZE * config.QUERY_RECORD_DURATION)
                        for _ in range(num_chunks):
                            data = stream.read(config.CHUNK_SIZE, exception_on_overflow=False)
                            frames.append(data)
                        
                        stream.stop_stream()
                        stream.close()
                        
                        audio_bytes = b''.join(frames)
                        
                        # Apply AGC (Automatic Gain Control) to normalize volume
                        try:
                            import numpy as np
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            # Check audio energy
                            rms = np.sqrt(np.mean(audio_array ** 2))
                            max_amplitude = np.max(np.abs(audio_array))
                            print(f"[WakeWord] Audio stats: RMS={rms:.4f}, Max={max_amplitude:.4f}")
                            
                            # Apply AGC if audio is too quiet (target RMS ~0.15)
                            target_rms = 0.15
                            if rms > 0.001:  # Only if there's some audio
                                gain = min(target_rms / rms, 10.0)  # Limit gain to 10x
                                audio_array = np.clip(audio_array * gain, -1.0, 1.0)
                                print(f"[WakeWord] Applied AGC: gain={gain:.2f}x")
                            else:
                                print(f"[WakeWord] ⚠️ Audio is very quiet (RMS < 0.001), may be silence")
                            
                            # Convert back to bytes
                            audio_bytes = (audio_array * 32768.0).astype(np.int16).tobytes()
                        except Exception as agc_error:
                            print(f"[WakeWord] AGC failed: {agc_error}, using raw audio")
                        
                        # Send to voice WebSocket
                        if audio_bytes and self.voice_ws:
                            print(f"[WakeWord] 📤 Sending {len(audio_bytes)} bytes to cloud...")
                            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                            
                            try:
                                await self.voice_ws.send(json.dumps({
                                    "type": "audio_chunk",
                                    "encoding": "base64",
                                    "data": audio_b64
                                }))
                                await self.voice_ws.send(json.dumps({
                                    "type": "audio_end",
                                    "encoding": "base64",
                                    "sampleRate": self.audio_sample_rate
                                }))
                                print("[WakeWord] ✅ Audio sent to cloud for processing")
                            except Exception as e:
                                if "closed" in str(e).lower() or "connection" in str(e).lower():
                                    print("[WakeWord] Reconnecting to voice endpoint...")
                                    await self.connect_voice_websocket()
                                else:
                                    print(f"[WakeWord] Error sending audio: {e}")
                    except Exception as e:
                        print(f"[WakeWord] Error recording query: {e}")
                
                # Small delay before next detection cycle
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"[WakeWord] Error in detection loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)
    
    async def receive_voice_responses(self):
        """Receive responses from voice WebSocket (transcripts, AI responses, etc.)."""
        while self.running:
            try:
                if not self.voice_ws:
                    await asyncio.sleep(1)
                    continue
                
                try:
                    message = await asyncio.wait_for(self.voice_ws.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    msg_type = data.get('type', '')
                    
                    if msg_type == 'transcript':
                        transcript = data.get('text', '')
                        print(f"[Voice] Transcript: {transcript}")
                        # Check for wake word in transcript
                        transcript_lower = transcript.lower()
                        if 'rovy' in transcript_lower and not self.is_recording:
                            print("[Voice] ✅ Wake word detected in transcript!")
                            # Extract query after wake word
                            query = transcript_lower
                            for wake in ['rovy', 'hey rovy']:
                                query = query.replace(wake, '').strip()
                            
                            # If there's a query after wake word, start recording to capture full query
                            if query:
                                print(f"[Voice] Query detected: '{query}', starting full recording...")
                                await self.record_and_stream_audio(duration=10.0)
                            else:
                                # Just wake word, start recording for user's next command
                                print("[Voice] Wake word only, starting recording for command...")
                                await self.record_and_stream_audio(duration=10.0)
                    
                    elif msg_type == 'response':
                        response = data.get('text', '')
                        print(f"[Voice] 🤖 AI Response: {response}")
                    
                    elif msg_type == 'status':
                        status = data.get('message', '')
                        print(f"[Voice] Status: {status}")
                        
                        # Speak important status messages to user
                        if 'could not transcribe' in status.lower():
                            await asyncio.to_thread(self.speak_acknowledgment, "Sorry, I didn't catch that. Please speak clearly after the beep.")
                        elif 'no audio' in status.lower():
                            await asyncio.to_thread(self.speak_acknowledgment, "I didn't hear anything. Please try again.")
                        elif 'not available' in status.lower():
                            await asyncio.to_thread(self.speak_acknowledgment, "Voice service is not available right now.")
                    
                except asyncio.TimeoutError:
                    continue  # No message, continue waiting
                except websockets.exceptions.ConnectionClosed:
                    print("[Voice] Connection closed, reconnecting...")
                    await self.connect_voice_websocket()
                except Exception as e:
                    print(f"[Voice] Error receiving message: {e}")
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                print(f"[Voice] Receive loop error: {e}")
                await asyncio.sleep(1.0)
    
    def ensure_oakd_initialized(self):
        """Lazy-load OAK-D camera only when needed."""
        if self.oakd_device is not None and self.oakd_queue is not None:
            return True  # Already initialized
        
        print("[Camera] Lazy-loading OAK-D camera for vision command...")
        if not DEPTHAI_OK:
            print("[Camera] ERROR: DepthAI not available")
            return False
        
        try:
            available = dai.Device.getAllAvailableDevices()
            if not available:
                print("[Camera] ERROR: No OAK-D devices found")
                return False
            
            print(f"[Camera] Found {len(available)} OAK-D device(s)")
            
            # Create OAK-D pipeline
            pipeline = dai.Pipeline()
            
            # Create color camera node
            if hasattr(dai, "node") and hasattr(dai.node, "ColorCamera"):
                camera = pipeline.create(dai.node.ColorCamera)
            else:
                camera = pipeline.createColorCamera()
            
            camera.setPreviewSize(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
            camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            camera.setInterleaved(False)
            camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            camera.setFps(30)
            
            # Create XLinkOut for preview
            if hasattr(dai, "node") and hasattr(dai.node, "XLinkOut"):
                xout = pipeline.create(dai.node.XLinkOut)
            else:
                xout = pipeline.createXLinkOut()
            xout.setStreamName("preview")
            camera.preview.link(xout.input)
            
            # Start pipeline
            self.oakd_device = dai.Device(pipeline)
            self.oakd_queue = self.oakd_device.getOutputQueue("preview", maxSize=4, blocking=False)
            
            # Test capture
            time.sleep(0.5)
            frame_data = self.oakd_queue.tryGet()
            if frame_data is not None:
                frame = frame_data.getCvFrame()
                if frame is not None:
                    self.camera_type = 'oakd'
                    print("[Camera] OAK-D camera initialized successfully")
                    return True
            
            # Failed - clean up
            self.oakd_device.close()
            self.oakd_device = None
            self.oakd_queue = None
            print("[Camera] ERROR: OAK-D test capture failed")
            return False
            
        except Exception as e:
            print(f"[Camera] ERROR: OAK-D initialization failed: {e}")
            if self.oakd_device:
                try:
                    self.oakd_device.close()
                except:
                    pass
                self.oakd_device = None
                self.oakd_queue = None
            return False
    
    def _reinit_oakd(self):
        """Reinitialize OAK-D camera after error."""
        # Prevent too frequent reinitializations (max once per 5 seconds)
        if time.time() - self.oakd_last_reinit < 5:
            return False
        
        self.oakd_last_reinit = time.time()
        print("[Camera] Attempting to reinitialize OAK-D camera...")
        
        # Close existing device
        if self.oakd_device:
            try:
                self.oakd_device.close()
            except:
                pass
            self.oakd_device = None
            self.oakd_queue = None
        
        # Wait a bit for device to be released
        time.sleep(1.0)
        
        # Try to reinitialize OAK-D only
        if not DEPTHAI_OK:
            return False
        
        try:
            available = dai.Device.getAllAvailableDevices()
            if not available:
                print("[Camera] No OAK-D devices available for reinit")
                return False
            
            # Create OAK-D pipeline
            pipeline = dai.Pipeline()
            
            # Create color camera node
            if hasattr(dai, "node") and hasattr(dai.node, "ColorCamera"):
                camera = pipeline.create(dai.node.ColorCamera)
            else:
                camera = pipeline.createColorCamera()
            
            camera.setPreviewSize(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
            camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            camera.setInterleaved(False)
            camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            manual_control_fps = 30
            camera.setFps(manual_control_fps)
            
            # Create XLinkOut for preview
            if hasattr(dai, "node") and hasattr(dai.node, "XLinkOut"):
                xout = pipeline.create(dai.node.XLinkOut)
            else:
                xout = pipeline.createXLinkOut()
            xout.setStreamName("preview")
            camera.preview.link(xout.input)
            
            # Start pipeline
            self.oakd_device = dai.Device(pipeline)
            self.oakd_queue = self.oakd_device.getOutputQueue("preview", maxSize=4, blocking=False)
            
            # Test capture
            time.sleep(0.5)
            frame_data = self.oakd_queue.tryGet()
            if frame_data is not None:
                frame = frame_data.getCvFrame()
                if frame is not None:
                    self.camera_type = 'oakd'
                    print("[Camera] OAK-D camera reinitialized successfully")
                    return True
            
            # If test failed, close device
            self.oakd_device.close()
            self.oakd_device = None
            self.oakd_queue = None
            return False
            
        except Exception as e:
            print(f"[Camera] OAK-D reinit failed: {e}")
            if self.oakd_device:
                try:
                    self.oakd_device.close()
                except:
                    pass
                self.oakd_device = None
                self.oakd_queue = None
            return False
    
    def capture_image(self, prefer_oakd: bool = False) -> Optional[bytes]:
        """
        Capture image from camera as JPEG bytes.
        
        Args:
            prefer_oakd: If True, use OAK-D if initialized. If False (default), use USB camera.
        """
        frame = None
        
        # Prefer USB camera by default (more reliable, less resource intensive)
        if not prefer_oakd and self.usb_camera and CAMERA_OK:
            ret, frame = self.usb_camera.read()
            if ret and frame is not None:
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                self.last_image = buffer.tobytes()
                self.last_image_time = time.time()
                return self.last_image
        
        # Fallback to OAK-D only if USB camera failed or prefer_oakd is True
        if self.oakd_queue is not None:
            try:
                # Get the latest frame (drop old ones if queue has multiple)
                frame_data = None
                while True:
                    new_frame = self.oakd_queue.tryGet()
                    if new_frame is None:
                        break
                    frame_data = new_frame  # Keep the latest frame
                
                if frame_data is not None:
                    frame = frame_data.getCvFrame()
                    self.oakd_error_count = 0  # Reset error count on success
            except Exception as e:
                error_str = str(e)
                # Check if it's an X_LINK_ERROR (communication error)
                if 'X_LINK_ERROR' in error_str or 'Communication exception' in error_str:
                    self.oakd_error_count += 1
                    if self.oakd_error_count >= 3:  # After 3 consecutive errors, try to reinit
                        print(f"[Camera] OAK-D communication error detected, attempting recovery...")
                        if self._reinit_oakd():
                            self.oakd_error_count = 0
                        else:
                            print(f"[Camera] OAK-D reinitialization failed")
                    else:
                        print(f"[Camera] OAK-D capture error ({self.oakd_error_count}/3): {error_str[:100]}")
                else:
                    print(f"[Camera] OAK-D capture error: {error_str[:100]}")
                return None
        
        if frame is None:
            return None
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        self.last_image = buffer.tobytes()
        self.last_image_time = time.time()
        return self.last_image
    
    def get_cached_image(self) -> Optional[bytes]:
        """Get last captured image (for efficiency)."""
        # For OAK-D, always get fresh frame for better FPS (cache only 0.02s = 50 FPS max)
        # For USB camera, cache longer to reduce load
        cache_time = 0.02 if self.camera_type == 'oakd' else 0.1
        max_stale_time = 0.5  # Don't return images older than 500ms (indicates camera failure)
        
        if self.last_image and (time.time() - self.last_image_time) < cache_time:
            return self.last_image
        
        # If cached image is too stale, don't return it (camera likely failed)
        if self.last_image and (time.time() - self.last_image_time) > max_stale_time:
            self.last_image = None  # Clear stale cache
        
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
    
    async def detect_gesture_from_frame(self, image_bytes: bytes):
        """Detect gesture from OAK-D camera frame using cloud server."""
        try:
            # Try aiohttp first (async, better for this use case)
            try:
                import aiohttp
                aiohttp_available = True
            except ImportError:
                aiohttp_available = False
            
            # Cloud server HTTP URL (not WebSocket)
            cloud_http_url = config.SERVER_URL.replace('ws://', 'http://').replace('wss://', 'https://')
            cloud_http_url = cloud_http_url.replace(':8765', ':8000')  # FastAPI runs on 8000
            
            if aiohttp_available:
                # Use aiohttp for async HTTP requests
                form_data = aiohttp.FormData()
                form_data.add_field('file', 
                                  image_bytes,
                                  filename='gesture.jpg',
                                  content_type='image/jpeg')
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{cloud_http_url}/gesture/detect",
                        data=form_data,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            gesture = result.get('gesture', 'none')
                            confidence = result.get('confidence', 0.0)
                            
                            if confidence > 0.5:
                                self.current_gesture = gesture
                                self.gesture_confidence = confidence
                                print(f"[Gesture] Detected: {gesture} (confidence: {confidence:.2f})")
                            else:
                                self.current_gesture = 'none'
                                self.gesture_confidence = 0.0
                        else:
                            self.current_gesture = 'none'
            else:
                # Fallback to requests (sync, but works)
                import requests
                files = {'file': ('gesture.jpg', image_bytes, 'image/jpeg')}
                response = requests.post(
                    f"{cloud_http_url}/gesture/detect",
                    files=files,
                    timeout=5
                )
                if response.status_code == 200:
                    result = response.json()
                    gesture = result.get('gesture', 'none')
                    confidence = result.get('confidence', 0.0)
                    
                    if confidence > 0.5:
                        self.current_gesture = gesture
                        self.gesture_confidence = confidence
                        print(f"[Gesture] Detected: {gesture} (confidence: {confidence:.2f})")
                    else:
                        self.current_gesture = 'none'
                        self.gesture_confidence = 0.0
                else:
                    self.current_gesture = 'none'
        except Exception as e:
            # Log errors but don't spam - only log every 10th error
            if not hasattr(self, '_gesture_error_count'):
                self._gesture_error_count = 0
            self._gesture_error_count += 1
            if self._gesture_error_count % 10 == 1:
                print(f"[Gesture] Detection error (logged every 10th): {e}")
            self.current_gesture = 'none'
            self.gesture_confidence = 0.0
    
    async def stream_to_cloud(self):
        """Stream audio/video to cloud for AI processing."""
        if not self.ws:
            return
        
        print("[Cloud Stream] Starting...")
        
        image_interval = 1.0 / config.CAMERA_FPS
        sensor_interval = 5.0
        gesture_interval = 0.5  # Detect gestures every 500ms
        
        last_image_time = 0
        last_sensor_time = 0
        last_gesture_time = 0
        
        while self.running and self.ws:
            try:
                now = time.time()
                
                # Send image periodically - use USB camera for cloud streaming
                if CAMERA_OK and self.usb_camera and (now - last_image_time) >= image_interval:
                    ret, frame = self.usb_camera.read()
                    if ret and frame is not None:
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                        image_bytes = buffer.tobytes()
                        await self.ws.send(json.dumps({
                            "type": "image_data",
                            "image_base64": base64.b64encode(image_bytes).decode('utf-8'),
                            "width": config.CAMERA_WIDTH,
                            "height": config.CAMERA_HEIGHT,
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                    last_image_time = now
                
                # Gesture detection DISABLED in cloud streaming - only enable on-demand
                # Gestures will be detected when explicitly requested via vision commands
                
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
        
        # Release USB camera separately if it's different from self.camera
        if self.usb_camera and self.usb_camera != self.camera:
            try:
                self.usb_camera.release()
            except:
                pass
            self.usb_camera = None
        
        if self.oakd_device:
            try:
                self.oakd_device.close()
            except:
                pass
            self.oakd_device = None
            self.oakd_queue = None
        
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
# This prevents overdrive/clipping at max volume (USB speaker HK-5008 clips at 100%)
HW_VOLUME_MAX = 125  # 85% of 147

def detect_audio_card() -> Optional[int]:
    """Auto-detect USB audio card with PCM control.
    
    Returns:
        Card number (0-31) if found, None otherwise
    """
    try:
        # First, try to list all cards
        result = subprocess.run(
            ['aplay', '-l'],
            capture_output=True,
            text=True,
            timeout=3
        )
        if result.returncode != 0:
            return None
        
        # Parse output to find USB audio devices
        usb_cards = []
        for line in result.stdout.split('\n'):
            if 'card' in line and 'USB Audio' in line:
                # Exclude camera audio devices
                if 'Camera' not in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'card' and i + 1 < len(parts):
                            try:
                                card_num = int(parts[i + 1].rstrip(':'))
                                usb_cards.append(card_num)
                            except ValueError:
                                continue
        
        # Test each USB card to see if it has PCM control
        for card_num in usb_cards:
            try:
                test_result = subprocess.run(
                    ['amixer', '-c', str(card_num), 'sget', 'PCM'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if test_result.returncode == 0 and 'Playback' in test_result.stdout:
                    print(f"[Volume] Auto-detected USB audio card: {card_num}")
                    return card_num
            except Exception:
                continue
        
        # Fallback: try cards 0-7 for PCM control
        for card_num in range(8):
            try:
                test_result = subprocess.run(
                    ['amixer', '-c', str(card_num), 'sget', 'PCM'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if test_result.returncode == 0 and 'Playback' in test_result.stdout:
                    print(f"[Volume] Auto-detected audio card with PCM: {card_num}")
                    return card_num
            except Exception:
                continue
        
        return None
    except Exception as e:
        print(f"[Volume] Error detecting audio card: {e}")
        return None

# Auto-detect audio card on startup
HW_CARD = detect_audio_card()
if HW_CARD is None:
    print("[Volume] ⚠️  Could not detect audio card, volume control may not work")
    HW_CARD = 2  # Fallback to card 2 (common USB audio)

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
    
    # Include current gesture detection result
    status_data["gesture"] = robot.current_gesture
    status_data["gesture_confidence"] = robot.gesture_confidence
    
    return status_data


@app.get("/shot")
async def get_shot():
    """Get single camera frame - uses USB camera for manual control."""
    robot = get_robot()
    
    # Use USB camera for manual control (faster and more reliable)
    if robot.usb_camera is None or not robot.usb_camera.isOpened():
        raise HTTPException(status_code=503, detail="USB camera not available")
    
    # Capture from USB camera
    ret, frame = robot.usb_camera.read()
    if ret and frame is not None:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        image_bytes = buffer.tobytes()
    else:
        image_bytes = None
    
    # If capture failed (queue empty), try to use cached image as fallback
    if not image_bytes:
        # First try get_cached_image() which respects cache time
        image_bytes = robot.get_cached_image()
        
        # If that also fails, try using last_image directly (even if slightly stale)
        # This helps when requests come faster than camera FPS
        if not image_bytes and robot.last_image:
            max_stale_time = 0.5  # Use cached image up to 500ms old
            if (time.time() - robot.last_image_time) < max_stale_time:
                image_bytes = robot.last_image
                # Log that we're using stale cached image (only occasionally to avoid spam)
                if not hasattr(robot, '_stale_fallback_count'):
                    robot._stale_fallback_count = 0
                robot._stale_fallback_count += 1
                if robot._stale_fallback_count % 20 == 1:
                    age_ms = int((time.time() - robot.last_image_time) * 1000)
                    print(f"[API] /shot using stale cached image ({age_ms}ms old, fallback #{robot._stale_fallback_count})")
    
    # If both capture and cache failed, return error
    if not image_bytes:
        raise HTTPException(status_code=503, detail="OAK-D camera frame not available (queue empty)")
    
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/snapshot")
async def get_snapshot():
    """Get single camera frame from USB camera."""
    robot = get_robot()
    
    # Check if USB camera is connected and available
    if not CAMERA_OK:
        raise HTTPException(status_code=503, detail="OpenCV not available")
    
    if robot.usb_camera is None:
        raise HTTPException(status_code=503, detail="USB camera not connected")
    
    # Verify camera is actually opened
    if not robot.usb_camera.isOpened():
        raise HTTPException(status_code=503, detail="USB camera is not opened")
    
    # Capture image directly from USB camera
    try:
        ret, frame = robot.usb_camera.read()
        if not ret or frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame from USB camera")
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        image_bytes = buffer.tobytes()
        
        return Response(content=image_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to capture image from USB camera: {str(e)}")


@app.get("/video")
async def video_stream():
    """MJPEG video stream - uses USB camera for mobile app."""
    robot = get_robot()
    
    # Use USB camera for mobile app streaming
    if robot.usb_camera is None or not robot.usb_camera.isOpened():
        raise HTTPException(status_code=503, detail="USB camera not available")
    
    # Use 30 FPS for mobile app
    stream_fps = 30
    frame_interval = 1.0 / stream_fps
    
    def generate_frames():
        consecutive_failures = 0
        max_failures = 30  # Allow more failures for MJPEG (1 second at 30fps)
        
        while True:
            start_time = time.time()
            
            # Capture directly from USB camera
            ret, frame = robot.usb_camera.read()
            if ret and frame is not None:
                # Encode to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                image_bytes = buffer.tobytes()
                
                consecutive_failures = 0  # Reset on success
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"[API] MJPEG stream failed {max_failures} times, ending stream")
                    break
            
            # Sleep only the remaining time to maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/camera/ws")
async def camera_websocket(websocket: WebSocket):
    """WebSocket camera stream (base64 JPEG) - uses USB camera for mobile app."""
    robot = get_robot()
    
    # Use USB camera for mobile app streaming (separate from OAK-D used for navigation)
    if robot.usb_camera is None or not robot.usb_camera.isOpened():
        await websocket.accept()
        await websocket.close(code=1011, reason="USB camera not available")
        return
    
    await websocket.accept()
    print("[API] Camera WebSocket connected (USB Camera)")
    
    # Use 30 FPS for mobile app
    stream_fps = 30
    frame_interval = 1.0 / stream_fps
    
    try:
        consecutive_failures = 0
        max_failures = 50  # Increased tolerance - camera may take time to stabilize
        
        while True:
            start_time = time.time()
            
            # Check if navigation is active (OAK-D using USB bandwidth)
            if robot.is_navigating:
                # Send a status message instead of camera frames during navigation
                # This prevents USB bandwidth contention
                await websocket.send_json({
                    "status": "navigation_active",
                    "message": "Camera paused during autonomous navigation",
                    "timestamp": time.time()
                })
                await asyncio.sleep(1.0)  # Update status every second
                continue
            
            # Capture directly from USB camera
            ret, frame = robot.usb_camera.read()
            if ret and frame is not None:
                # Encode to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                image_bytes = buffer.tobytes()
                
                consecutive_failures = 0  # Reset on success
                await websocket.send_json({
                    "frame": base64.b64encode(image_bytes).decode('utf-8'),
                    "timestamp": time.time(),
                    "gesture": robot.current_gesture,
                    "gesture_confidence": robot.gesture_confidence
                })
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"[API] Camera stream failed {max_failures} times, closing connection")
                    await websocket.close(code=1011, reason="Camera capture failed")
                    break
                # Add small delay on failure to avoid tight loop
                await asyncio.sleep(0.1)
            
            # Sleep only the remaining time to maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    except WebSocketDisconnect:
        print("[API] Camera WebSocket disconnected")
    except Exception as e:
        print(f"[API] Camera WebSocket error: {e}")


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


@app.post("/navigation")
async def control_navigation(command: NavigationCommand):
    """Control autonomous navigation with OAK-D camera."""
    robot = get_robot()
    
    print(f"[Navigation API] {command.action}")
    
    # Import navigator dynamically when needed
    if not hasattr(robot, 'navigator'):
        try:
            import sys
            import os
            # Add oakd_navigation to path
            nav_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'oakd_navigation')
            if nav_path not in sys.path:
                sys.path.insert(0, nav_path)
            
            from rovy_integration import RovyNavigator
            robot.navigator = None  # Will be initialized on start
            print("[Navigation API] Module loaded")
        except Exception as e:
            print(f"[Navigation API] Failed to import: {e}")
            raise HTTPException(status_code=500, detail=f"Navigation not available: {e}")
    
    # Handle different navigation actions
    if command.action == 'start_explore':
        def start_nav():
            try:
                robot.is_navigating = True  # Pause USB camera to avoid bandwidth contention
                print("[Navigation API] USB camera paused (OAK-D navigation active)")
                
                if robot.navigator is None:
                    from rovy_integration import RovyNavigator
                    robot.navigator = RovyNavigator(rover_instance=robot.rover)
                    robot.navigator.start()
                
                print(f"[Navigation API] Starting exploration (duration={command.duration})")
                robot.navigator.explore(duration=command.duration)
            except Exception as e:
                print(f"[Navigation API] Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                robot.is_navigating = False  # Resume USB camera
                print("[Navigation API] USB camera resumed")
        
        # Run navigation in separate thread
        threading.Thread(target=start_nav, daemon=True).start()
        return {"status": "started", "action": "explore"}
    
    elif command.action == 'stop':
        if hasattr(robot, 'navigator') and robot.navigator:
            def stop_nav():
                try:
                    robot.navigator.stop()
                    robot.navigator.cleanup()
                    robot.navigator = None
                except Exception as e:
                    print(f"[Navigation API] Stop error: {e}")
                finally:
                    robot.is_navigating = False  # Resume USB camera
                    print("[Navigation API] USB camera resumed")
            
            threading.Thread(target=stop_nav, daemon=True).start()
            return {"status": "stopped", "action": "stop"}
        else:
            robot.is_navigating = False  # Ensure flag is cleared
            return {"status": "already_stopped"}
    
    elif command.action == 'goto':
        x = command.x
        y = command.y
        
        if x is None or y is None:
            raise HTTPException(status_code=400, detail="x and y coordinates required for goto")
        
        def navigate_to():
            try:
                robot.is_navigating = True  # Pause USB camera
                print("[Navigation API] USB camera paused (OAK-D navigation active)")
                
                if robot.navigator is None:
                    from rovy_integration import RovyNavigator
                    robot.navigator = RovyNavigator(rover_instance=robot.rover)
                    robot.navigator.start()
                
                print(f"[Navigation API] Navigating to ({x}, {y})")
                robot.navigator.navigate_to(x, y)
            except Exception as e:
                print(f"[Navigation API] Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                robot.is_navigating = False  # Resume USB camera
                print("[Navigation API] USB camera resumed")
        
        threading.Thread(target=navigate_to, daemon=True).start()
        return {"status": "started", "action": "goto", "target": {"x": x, "y": y}}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {command.action}")


@app.post("/speak")
async def speak_text(request: dict):
    """Speak text using robot's TTS (from cloud AI response) with language support."""
    robot = get_robot()
    
    text = request.get("text", "")
    audio_base64 = request.get("audio", None)
    language = request.get("language", "en")  # Get language for Piper voice selection
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    if language != "en":
        print(f"[Speak] ({language}) {text[:80]}...")
    else:
        print(f"[Speak] {text[:80]}...")
    
    # Run TTS in background thread so it doesn't block API response
    def do_speak():
        try:
            import tempfile
            import os
            
            robot.is_speaking = True  # Pause wake word detection
            
            # Select Piper voice based on language
            piper_voice = config.PIPER_VOICES.get(language, config.PIPER_VOICES.get("en"))
            
            # Check if voice file exists
            if not os.path.exists(piper_voice):
                print(f"[Speak] Warning: Piper voice not found for {language}: {piper_voice}")
                # Fall back to English voice
                piper_voice = config.PIPER_VOICES.get("en")
                if not os.path.exists(piper_voice):
                    print(f"[Speak] Error: No Piper voices available")
                    robot.is_speaking = False
                    return
                print(f"[Speak] Using English voice as fallback")
            
            # Create temp wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                wav_path = f.name
            
            # Generate speech with Piper
            print(f"[Speak] Generating with Piper ({os.path.basename(piper_voice)})...")
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
                
                # Use aplay with correct device - most reliable on Pi
                # -D specifies the device, plughw:2,0 for USB speaker (Card 2)
                play_proc = subprocess.run(
                    ['aplay', '-D', config.SPEAKER_DEVICE, wav_path],
                    capture_output=True,
                    timeout=30
                )
                
                if play_proc.returncode == 0:
                    print(f"[Speak] Played successfully with aplay on {config.SPEAKER_DEVICE}")
                else:
                    # Fallback to ffplay (but may not use correct device)
                    print(f"[Speak] aplay failed, trying ffplay...")
                    subprocess.run(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', wav_path], capture_output=True, timeout=30)
                    print(f"[Speak] Played with ffplay fallback")
                
                os.unlink(wav_path)
            else:
                error = proc.stderr.decode()[:200] if proc.stderr else "Unknown error"
                print(f"[Speak] Piper failed: {error}")
                
        except Exception as e:
            print(f"[Speak] TTS error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Wait 0.5s for audio to clear from mic before resuming detection
            import time
            time.sleep(0.5)
            robot.is_speaking = False  # Resume wake word detection
    
    # Start speaking in background
    threading.Thread(target=do_speak, daemon=True).start()
    
    # Return immediately
    return {"status": "speaking", "method": "piper_async", "language": language}


@app.post("/dance")
async def trigger_dance(request: dict):
    """Trigger a dance routine on the robot.
    
    Request body:
        style: Dance style - 'party', 'wiggle', or 'spin' (optional, default: 'party')
        duration: Duration in seconds (optional, default: 10)
        with_music: Play music during dance (optional, default: False)
        music_genre: Music genre if with_music is True (optional, default: 'dance')
    """
    robot = get_robot()
    
    if not robot.rover:
        raise HTTPException(status_code=503, detail="Rover not connected")
    
    style = request.get("style", "party")
    duration = request.get("duration", 10)
    with_music = request.get("with_music", False)
    music_genre = request.get("music_genre", "dance")
    
    # Validate style
    valid_styles = ['party', 'wiggle', 'spin']
    if style not in valid_styles:
        raise HTTPException(status_code=400, detail=f"Invalid style. Must be one of: {valid_styles}")
    
    # Validate duration
    if not isinstance(duration, (int, float)) or duration <= 0 or duration > 60:
        raise HTTPException(status_code=400, detail="Duration must be between 0 and 60 seconds")
    
    print(f"[Dance] 💃 Starting {style} dance for {duration}s" + (f" with {music_genre} music" if with_music else ""))
    
    # Run dance in background thread
    def do_dance():
        try:
            # Display on OLED
            robot.rover.display_lines([
                "DANCE MODE" + (" 🎵" if with_music else ""),
                f"Style: {style}",
                f"Time: {duration}s",
                "💃🕺💃"
            ])
            
            # Start music if requested
            music_started = False
            if with_music:
                try:
                    import requests
                    print(f"[Dance] 🎵 Starting {music_genre} music...")
                    response = requests.post(
                        "http://localhost:8000/music",
                        json={"action": "play", "genre": music_genre},
                        timeout=5
                    )
                    if response.status_code == 200:
                        music_started = True
                        import time
                        time.sleep(2)  # Let music start
                    else:
                        print("[Dance] ⚠️ Music player not available, dancing without music")
                except Exception as music_error:
                    print(f"[Dance] ⚠️ Music error: {music_error}, dancing without music")
            
            # Perform dance
            robot.rover.dance(style=style, duration=duration)
            
            # Stop music after dance
            if music_started:
                try:
                    import requests
                    print("[Dance] 🎵 Stopping music...")
                    requests.post(
                        "http://localhost:8000/music",
                        json={"action": "stop"},
                        timeout=5
                    )
                except Exception as stop_error:
                    print(f"[Dance] ⚠️ Failed to stop music: {stop_error}")
            
            # Reset display after dance
            robot.rover.display_lines([
                "ROVY",
                "Ready",
                "",
                ""
            ])
        except Exception as e:
            print(f"[Dance] Error: {e}")
            import traceback
            traceback.print_exc()
    
    threading.Thread(target=do_dance, daemon=True).start()
    
    return {
        "status": "ok",
        "message": f"Dancing {style} style for {duration} seconds" + (" with music" if with_music else ""),
        "style": style,
        "duration": duration,
        "with_music": with_music,
        "music_genre": music_genre if with_music else None
    }


# Global music player state
music_player_process = None
music_paused = False
current_search_term = None
music_track_index = 0  # Track which result to play from search

@app.post("/music")
async def control_music_simple(request: dict):
    """Control music playback using YouTube Music API (new system).
    
    Request body:
        action: 'play' or 'stop' (required)
        genre: Music genre for 'play' action (optional, default: 'dance')
                Options: 'dance', 'party', 'classical', 'jazz', 'rock', 'pop', 'chill', 'electronic', 'fun'
    """
    robot = get_robot()
    
    action = request.get("action", "play")
    genre = request.get("genre", "dance")
    
    # Validate action
    if action not in ['play', 'stop', 'status']:
        raise HTTPException(status_code=400, detail="Action must be 'play', 'stop', or 'status'")
    
    # Validate genre
    valid_genres = ['dance', 'party', 'classical', 'jazz', 'rock', 'pop', 'chill', 'electronic', 'fun']
    if action == 'play' and genre not in valid_genres:
        raise HTTPException(status_code=400, detail=f"Invalid genre. Must be one of: {valid_genres}")
    
    try:
        from music_player import get_music_player
        music_player = get_music_player()
        
        if not music_player or not music_player.yt_music:
            raise HTTPException(status_code=503, detail="YouTube Music not configured. Run auth_youtube.py on the robot.")
        
        if action == 'play':
            print(f"[Music] 🎵 Playing {genre} music")
            
            if robot.rover:
                robot.rover.display_lines([
                    "MUSIC MODE",
                    f"Genre: {genre}",
                    "Loading...",
                    "🎵"
                ])
            
            def play_music():
                success = music_player.play_random(genre)
                if success and robot.rover:
                    song = music_player.current_song
                    if song:
                        robot.rover.display_lines([
                            "NOW PLAYING",
                            song['title'][:21],
                            song['artist'][:21],
                            "🎵"
                        ])
                elif robot.rover:
                    robot.rover.display_lines([
                        "Music Error",
                        "No songs found",
                        f"Genre: {genre}",
                        ""
                    ])
            
            threading.Thread(target=play_music, daemon=True).start()
            
            return {
                "status": "ok",
                "action": "playing",
                "genre": genre
            }
        
        elif action == 'stop':
            print("[Music] ⏹️ Stopping music")
            music_player.stop()
            
            if robot.rover:
                robot.rover.display_lines([
                    "MUSIC",
                    "Stopped",
                    "",
                    ""
                ])
            
            return {
                "status": "ok",
                "action": "stopped"
            }
        
        elif action == 'status':
            status = music_player.get_status()
            return {
                "status": "ok",
                "is_playing": status['is_playing'],
                "current_song": status['current_song']
            }
    
    except ImportError:
        raise HTTPException(status_code=503, detail="Music player module not available")
    except Exception as exc:
        print(f"[Music] Error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Music control failed: {str(exc)}")


@app.post("/music/{action}")
async def control_music(action: str, request: dict = None):
    """Control music playback on robot speakers using YouTube/yt-dlp."""
    global music_player_process
    
    print(f"[Music] Action: {action}")
    
    valid_actions = ["play", "pause", "stop", "next", "previous"]
    if action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Use: {', '.join(valid_actions)}")
    
    # Get search query if provided
    query = request.get("query", "") if request else ""
    
    # Run music control in background
    def do_music_control():
        global music_player_process, music_paused, current_search_term, music_track_index
        
        try:
            if action == "play":
                # Resume if paused
                if music_paused and music_player_process and music_player_process.poll() is None:
                    import os
                    import signal
                    os.kill(music_player_process.pid, signal.SIGCONT)
                    music_paused = False
                    print(f"[Music] ✅ Resumed")
                    return
                
                # Stop any existing playback first
                if music_player_process and music_player_process.poll() is None:
                    music_player_process.terminate()
                    music_player_process.wait(timeout=2)
                
                # Determine search query
                if query:
                    search_term = query
                else:
                    # Default: popular music mix
                    search_term = "popular music mix 2024"
                
                current_search_term = search_term
                music_paused = False
                music_track_index = 0  # Reset track index for new search
                
                print(f"[Music] Searching YouTube for: {search_term}")
                
                # Use yt-dlp to search and get audio stream URL
                # --get-url returns the direct media URL without downloading
                result = subprocess.run(
                    ['yt-dlp', '--default-search', 'ytsearch1:', 
                     '-f', 'bestaudio', '--get-url', '--no-warnings', 
                     '--quiet', search_term],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    audio_url = result.stdout.strip()
                    print(f"[Music] Found audio, starting playback...")
                    
                    # Play the audio URL with ffmpeg to speaker device
                    # Use ffmpeg to decode and aplay to output to correct device
                    music_player_process = subprocess.Popen(
                        f'ffmpeg -i "{audio_url}" -f wav -ac 2 -ar 44100 - 2>/dev/null | aplay -D {config.SPEAKER_DEVICE} -',
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    
                    print(f"[Music] ✅ Playing: {search_term}")
                else:
                    print(f"[Music] ❌ Could not find audio for: {search_term}")
            
            elif action == "pause":
                if music_player_process and music_player_process.poll() is None:
                    if not music_paused:
                        import os
                        import signal
                        os.kill(music_player_process.pid, signal.SIGSTOP)
                        music_paused = True
                        print(f"[Music] ✅ Paused")
                    else:
                        print(f"[Music] Already paused")
                else:
                    print(f"[Music] No music playing")
            
            elif action == "stop":
                if music_player_process and music_player_process.poll() is None:
                    music_player_process.terminate()
                    music_player_process.wait(timeout=2)
                    music_paused = False
                    print(f"[Music] ✅ Stopped")
                else:
                    print(f"[Music] No music playing")
            
            elif action == "next":
                # Play next song from similar search
                if current_search_term:
                    search_term = current_search_term
                else:
                    search_term = "popular music mix 2024"
                
                # Stop current playback
                if music_player_process and music_player_process.poll() is None:
                    music_player_process.terminate()
                    music_player_process.wait(timeout=2)
                
                music_paused = False
                music_track_index += 1  # Increment to get next track
                
                print(f"[Music] Searching for next song (#{music_track_index})...")
                
                # Search for multiple songs and pick one based on index
                # Use ytsearch5: to get 5 results, then pick based on track_index
                result = subprocess.run(
                    ['yt-dlp', '--default-search', 'ytsearch5:', 
                     '-f', 'bestaudio', '--get-url', '--no-warnings', 
                     '--quiet', '--playlist-items', str((music_track_index % 5) + 1),
                     search_term],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    audio_url = result.stdout.strip()
                    music_player_process = subprocess.Popen(
                        f'ffmpeg -i "{audio_url}" -f wav -ac 2 -ar 44100 - 2>/dev/null | aplay -D {config.SPEAKER_DEVICE} -',
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f"[Music] ✅ Next track playing")
                else:
                    print(f"[Music] ❌ Could not find next track")
            
            elif action == "previous":
                # For previous, go back to the previous track
                if current_search_term:
                    search_term = current_search_term
                else:
                    search_term = "popular music mix 2024"
                
                # Stop current playback
                if music_player_process and music_player_process.poll() is None:
                    music_player_process.terminate()
                    music_player_process.wait(timeout=2)
                
                music_paused = False
                music_track_index = max(0, music_track_index - 1)  # Decrement to get previous track
                
                print(f"[Music] Going to previous track (#{music_track_index})...")
                
                # Search for multiple songs and pick one based on index
                result = subprocess.run(
                    ['yt-dlp', '--default-search', 'ytsearch5:', 
                     '-f', 'bestaudio', '--get-url', '--no-warnings', 
                     '--quiet', '--playlist-items', str((music_track_index % 5) + 1),
                     search_term],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    audio_url = result.stdout.strip()
                    music_player_process = subprocess.Popen(
                        f'ffmpeg -i "{audio_url}" -f wav -ac 2 -ar 44100 - 2>/dev/null | aplay -D {config.SPEAKER_DEVICE} -',
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f"[Music] ✅ Previous track playing")
                else:
                    print(f"[Music] ❌ Could not find previous track")
                
        except Exception as e:
            print(f"[Music] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Execute in background
    threading.Thread(target=do_music_control, daemon=True).start()
    
    return {
        "status": "ok",
        "action": action,
        "message": f"Music {action} command sent"
    }


@app.post("/youtube-music/{action}")
async def control_youtube_music(action: str, request: dict):
    """Control YouTube Music playback on robot (from cloud AI)."""
    print(f"[YouTube Music] Action: {action}")
    
    valid_actions = ["play", "pause", "stop", "next", "previous"]
    if action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Use: {', '.join(valid_actions)}")
    
    query = request.get("query", "") if request else ""
    
    # Run YouTube Music control in background
    def do_youtube_music_control():
        try:
            import platform
            if platform.system() == "Linux":
                if action == "play":
                    # Open YouTube Music in chromium (headless browser on Pi)
                    # Play a random mix or user's library
                    print(f"[YouTube Music] Opening YouTube Music...")
                    
                    # Use chromium-browser to open YouTube Music
                    # The --app flag makes it fullscreen without browser UI
                    url = "https://music.youtube.com"
                    
                    subprocess.Popen([
                        'chromium-browser',
                        '--app=' + url,
                        '--autoplay-policy=no-user-gesture-required',
                        '--start-maximized'
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    print(f"[YouTube Music] ✅ Opened YouTube Music")
                
                else:
                    # For pause/next/previous, use playerctl (works with browser media)
                    print(f"[YouTube Music] Executing playerctl {action}")
                    result = subprocess.run(
                        ['playerctl', action],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        print(f"[YouTube Music] ✅ {action} successful")
                    else:
                        print(f"[YouTube Music] ⚠️ playerctl returned: {result.stderr}")
            else:
                print(f"[YouTube Music] ❌ Not supported on {platform.system()}")
                
        except Exception as e:
            print(f"[YouTube Music] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Execute in background
    threading.Thread(target=do_youtube_music_control, daemon=True).start()
    
    return {
        "status": "ok",
        "action": action,
        "message": f"YouTube Music {action} command sent"
    }


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
    global HW_CARD
    # Re-detect card in case it changed
    if HW_CARD is None:
        HW_CARD = detect_audio_card()
    
    # Try to read hardware volume and map back to 0-100 scale
    if HW_CARD is not None:
        try:
            result = subprocess.run(
                ['amixer', '-c', str(HW_CARD), 'get', 'PCM'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
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
    global HW_CARD
    # Re-detect card in case it changed
    if HW_CARD is None:
        HW_CARD = detect_audio_card()
    
    # Clamp volume to valid range
    volume = max(0, min(100, command.volume))
    AUDIO_STATE["volume"] = volume
    
    # Set hardware mixer volume with safe mapping
    # Map 0-100% to 0-125 (prevents overdrive at max volume)
    if HW_CARD is not None:
        try:
            hw_volume = int((volume / 100.0) * HW_VOLUME_MAX)
            result = subprocess.run(
                ['amixer', '-c', str(HW_CARD), 'sset', 'PCM', str(hw_volume)],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                hw_percent = int((hw_volume / 147.0) * 100)
                print(f"[Volume] Set to {volume}% (hardware: {hw_volume}/147 = {hw_percent}% on card {HW_CARD})")
            else:
                print(f"[Volume] Failed to set hardware volume: {result.stderr}")
                # Try to re-detect card
                HW_CARD = detect_audio_card()
                if HW_CARD is not None:
                    # Retry once
                    try:
                        result = subprocess.run(
                            ['amixer', '-c', str(HW_CARD), 'sset', 'PCM', str(hw_volume)],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if result.returncode == 0:
                            hw_percent = int((hw_volume / 147.0) * 100)
                            print(f"[Volume] Set to {volume}% (hardware: {hw_volume}/147 = {hw_percent}% on card {HW_CARD})")
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Volume] Set software volume to {volume}% (hardware control unavailable: {e})")
            # Try to re-detect card for next time
            HW_CARD = detect_audio_card()
    else:
        print(f"[Volume] Set software volume to {volume}% (no audio card detected)")
    
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
    
    # Initialize wake word detection
    robot_server.init_wake_word()
    
    # Try to connect to cloud (non-blocking)
    if WEBSOCKETS_OK:
        if await robot_server.connect_cloud():
            # Run cloud streaming in background
            asyncio.create_task(robot_server.stream_to_cloud())
            asyncio.create_task(robot_server.receive_from_cloud())
        else:
            print("[Cloud] Continuing without cloud connection")
    
    # Start wake word detection loop in background
    if robot_server.wake_word_enabled:
        asyncio.create_task(robot_server.wake_word_detection_loop())
        # Also start a task to receive responses from voice WebSocket
        asyncio.create_task(robot_server.receive_voice_responses())
    
    # Keep running
    try:
        while robot_server.running:
            await asyncio.sleep(1)
    except:
        robot_server.cleanup()


# Global uvicorn server instance for signal handler
uvicorn_server = None

def signal_handler(sig, frame):
    """Handle shutdown signals."""
    print("\n[Signal] Shutting down...")
    
    # Shutdown uvicorn gracefully
    if uvicorn_server:
        uvicorn_server.should_exit = True
    
    # Cleanup robot server
    if robot_server:
        robot_server.running = False
        robot_server.cleanup()
    
    # Give uvicorn a moment to shut down, then exit
    import time
    time.sleep(0.5)
    sys.exit(0)


def main():
    """Main entry point."""
    global uvicorn_server
    
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
    
    uvicorn_server = uvicorn.Server(config_uvicorn)
    
    # Add robot server startup to uvicorn lifespan
    async def startup():
        asyncio.create_task(run_robot_server())
    
    app.add_event_handler("startup", startup)
    
    # Run server
    uvicorn_server.run()


if __name__ == "__main__":
    main()

