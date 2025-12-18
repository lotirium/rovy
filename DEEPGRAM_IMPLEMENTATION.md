# Deepgram API Implementation Guide

This document explains the correct implementation of Deepgram's Python SDK for wake word detection on the Rovy robot.

## Overview

The Deepgram SDK provides real-time speech-to-text transcription through WebSocket streaming. We use it for wake word detection with the following improvements over the previous implementation:

## Key Fixes Applied

### 1. **Event Handler Scope Issue** (CRITICAL)

**Problem**: Event handlers were incorrectly defined as methods with `self` parameter inside the async method:

```python
# ❌ INCORRECT - handlers defined as methods inside a function
def on_message(self, result, **kwargs):
    sentence = result.channel.alternatives[0].transcript
    if is_final and self._check_wake_word(sentence):  # Won't work!
        self.wake_word_detected = True
```

**Solution**: Event handlers should be closures (nested functions) without `self` parameter:

```python
# ✅ CORRECT - handlers as closures accessing outer scope
def on_message(result, **kwargs):
    sentence = result.channel.alternatives[0].transcript
    if is_final and self._check_wake_word(sentence):  # Works via closure!
        self.wake_word_detected = True
```

The handlers access the outer `self` through Python's closure mechanism, not as method calls.

### 2. **Error Handling Improvements**

Added traceback printing to help debug issues:

```python
except Exception as e:
    print(f"[Deepgram] Error in on_message: {e}")
    import traceback
    traceback.print_exc()  # Show full error details
```

### 3. **SDK Installation**

The correct SDK version must be installed:

```bash
pip install "deepgram-sdk>=3.0,<4.0"
```

## File Structure

```
robot/
├── wake_word_detector_deepgram.py      # Main implementation (CORRECTED)
├── wake_word_detector_deepgram_v2.py   # Backup version (CORRECTED)
├── install_deepgram.sh                 # Installation script
├── test_deepgram.py                    # Test script
├── example_deepgram_wake_word.py       # Usage example
└── config.py                           # Configuration with API key
```

## Installation

### Step 1: Install Dependencies

```bash
cd /home/rovy/rovy_client/robot
./install_deepgram.sh
```

This installs:
- `deepgram-sdk` (v3.x)
- `pyaudio` (audio input)
- `numpy` and `scipy` (audio resampling)

### Step 2: Get API Key

1. Sign up at https://console.deepgram.com/signup (includes $200 free credit)
2. Create an API key in the console
3. Add to `config.py`:

```python
DEEPGRAM_API_KEY = "your_api_key_here"
```

### Step 3: Test Installation

```bash
python3 test_deepgram.py
```

This tests:
- ✅ SDK imports
- ✅ API connection
- ✅ Microphone detection

## Usage Example

```python
from wake_word_detector_deepgram import DeepgramWakeWordDetector
import asyncio
import config

async def main():
    # Initialize detector
    detector = DeepgramWakeWordDetector(
        api_key=config.DEEPGRAM_API_KEY,
        wake_words=["hey rovy", "rovy"],
        sample_rate=16000,
        device_sample_rate=44100,  # USB mic native rate
    )
    
    # Callback when wake word detected
    def on_wake_word(transcript):
        print(f"Wake word detected: {transcript}")
    
    # Listen for wake word
    detected = await detector.listen_for_wake_word_async(
        callback=on_wake_word,
        timeout=30.0  # Optional timeout
    )
    
    if detected:
        print(f"Transcript: {detector.last_transcript}")
    
    # Clean up
    detector.cleanup()

asyncio.run(main())
```

## Architecture

### Class: `DeepgramWakeWordDetector`

**Initialization Parameters:**
- `api_key`: Deepgram API key
- `wake_words`: List of wake words to detect (default: ["hey rovy"])
- `sample_rate`: Deepgram transcription rate (16000Hz recommended)
- `device_sample_rate`: Microphone native rate (44100Hz for USB mics)
- `device_index`: PyAudio device index (None = auto-detect)

**Main Method: `listen_for_wake_word_async()`**

This async method:
1. Creates Deepgram client and WebSocket connection
2. Opens PyAudio stream from microphone
3. Continuously streams audio to Deepgram
4. Processes transcription results via event handlers
5. Detects wake words with fuzzy matching
6. Returns `True` when wake word detected or `False` on timeout

**Event Handlers:**
- `on_message(result, **kwargs)`: Receives transcription results
- `on_error(error, **kwargs)`: Handles errors
- `on_close(close_msg, **kwargs)`: Handles connection closure

### Audio Pipeline

```
Microphone (44100Hz) 
    ↓
PyAudio Stream (100ms chunks)
    ↓
Resampling (44100Hz → 16000Hz) [if needed]
    ↓
Deepgram WebSocket
    ↓
Transcription Results (interim + final)
    ↓
Wake Word Detection (fuzzy matching)
    ↓
Callback + Return True
```

## Configuration Options

### Deepgram LiveOptions

Current configuration in the code:

```python
options = LiveOptions(
    model="nova-2",              # Latest model (fast + accurate)
    language="en-US",            # English (US)
    encoding="linear16",         # 16-bit PCM audio
    sample_rate=16000,           # 16kHz sample rate
    channels=1,                  # Mono audio
    punctuate=True,              # Add punctuation
    smart_format=True,           # Smart formatting
    interim_results=True,        # Get partial results
    utterance_end_ms="1000",     # 1s silence ends utterance
    vad_events=True,             # Voice activity detection
    endpointing=1000,            # 1s silence finalizes
)
```

### Wake Word Matching

The detector includes fuzzy matching for common misheard variations:

```python
# Exact matches
wake_words = ["hey rovy", "rovy", "hey robot", "hey"]

# Fuzzy matches (automatically detected)
variants = [
    "roevee", "rovee", "romy", "ruby", "rovie", "robbie",
    "roby", "robi", "rovey", "robey", "rovi", "rovvy",
    "rov me", "rob me", "rove", "rope"
]
```

## Integration with Rovy Robot

The detector is integrated into `main_api.py`. To enable Deepgram:

1. Set in `config.py`:
```python
USE_DEEPGRAM = True
DEEPGRAM_API_KEY = "your_api_key_here"
```

2. Restart the robot service:
```bash
sudo systemctl restart rovy.service
```

## Common Issues & Solutions

### Issue 1: "DeepgramClient has no attribute 'listen'"

**Cause**: Wrong SDK version installed

**Solution**:
```bash
pip uninstall deepgram-sdk
pip install "deepgram-sdk>=3.0,<4.0"
```

### Issue 2: Event handlers not working / AttributeError

**Cause**: Handlers defined as methods instead of closures

**Solution**: Use the corrected code (handlers without `self` parameter)

### Issue 3: "Failed to start connection"

**Causes**:
- Invalid API key
- Network issues
- Firewall blocking WebSocket

**Solution**:
- Verify API key in config.py
- Test with `python3 test_deepgram.py`
- Check internet connection

### Issue 4: No microphone detected

**Cause**: PyAudio can't find input device

**Solution**:
```bash
# List audio devices
python3 -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"

# Verify USB mic is connected
arecord -l
```

## API Costs

Deepgram pricing (as of 2025):
- **Nova-2 model**: $0.0043/minute (~$0.26/hour)
- **Free tier**: $200 credit (~773 hours of usage)
- **Wake word detection**: Only charged when actively listening

For a robot that listens continuously, costs are:
- **Per day (24hrs)**: ~$6.24
- **Per month**: ~$187

**Optimization tip**: Only activate listening when robot is awake/active to reduce costs.

## Performance

- **Latency**: ~200-500ms from speech to transcription
- **Accuracy**: >95% for clear speech (Nova-2 model)
- **CPU Usage**: Minimal (transcription done on Deepgram servers)
- **Network**: ~40 kbps upload bandwidth for 16kHz audio

## Advantages Over Local Detection

| Feature | Deepgram (Cloud) | Local (Whisper/Vosk) |
|---------|------------------|---------------------|
| Accuracy | Excellent (95%+) | Good (85-90%) |
| CPU Usage | Very Low | High (Pi struggles) |
| Latency | Low (200-500ms) | Medium-High (1-3s) |
| Internet Required | Yes | No |
| Cost | ~$0.26/hour | Free |
| Setup | Easy | Complex |

## Testing

Run the example script:
```bash
python3 example_deepgram_wake_word.py
```

Expected output:
```
============================================================
  Deepgram Wake Word Detector Example
============================================================

Wake words: hey rovy, rovy, hey robot, hey
The detector uses fuzzy matching for common misheard variations

Press Ctrl+C to stop

[Deepgram] Initializing with official SDK for wake words: ['hey rovy', 'rovy', 'hey robot', 'hey']
[Deepgram] 👂 Listening for: ['hey rovy', 'rovy', 'hey robot', 'hey']
[Deepgram] ✅ Connected to Deepgram streaming API (SDK v5)
[Deepgram] ✅ Audio stream opened at 44100Hz
[Deepgram] 💭 interim: 'hey'
[Deepgram] 💭 interim: 'hey ro'
[Deepgram] 📝 FINAL: 'hey rovy'
[Deepgram] ✅ Wake word detected!

🎉 WAKE WORD DETECTED!
   Transcript: 'hey rovy'

✅ Wake word was detected successfully!
```

## References

- **Deepgram Docs**: https://developers.deepgram.com/docs
- **Python SDK**: https://github.com/deepgram/deepgram-python-sdk
- **API Console**: https://console.deepgram.com/
- **Pricing**: https://deepgram.com/pricing

---

**Last Updated**: December 2025
**SDK Version**: 3.x
**Status**: ✅ Tested and Working

