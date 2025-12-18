# Deepgram Quick Start Guide

## TL;DR - Get Started in 3 Steps

### 1. Install Deepgram SDK

```bash
cd /home/rovy/rovy_client/robot
./install_deepgram.sh
```

### 2. Get API Key & Configure

1. Sign up: https://console.deepgram.com/signup (free $200 credit)
2. Create API key in console
3. Add to `robot/config.py`:

```python
DEEPGRAM_API_KEY = "your_api_key_here"
USE_DEEPGRAM = True
```

### 3. Test It

```bash
# Test installation
python3 robot/test_deepgram.py

# Try wake word detection
python3 robot/example_deepgram_wake_word.py
```

---

## What Was Fixed?

### Critical Bug: Event Handler Scope

**Before (BROKEN):**
```python
def on_message(self, result, **kwargs):  # ❌ Wrong!
    if is_final and self._check_wake_word(sentence):
        self.wake_word_detected = True
```

**After (FIXED):**
```python
def on_message(result, **kwargs):  # ✅ Correct!
    if is_final and self._check_wake_word(sentence):
        self.wake_word_detected = True
```

**Why?** Event handlers are closures (nested functions), not methods. They access the outer `self` through Python's closure mechanism.

---

## Quick Commands

```bash
# Install everything
./robot/install_deepgram.sh

# Test connection
python3 robot/test_deepgram.py

# Try wake word detection (say "hey rovy")
python3 robot/example_deepgram_wake_word.py

# Integrate with robot
sudo systemctl restart rovy.service

# Check robot logs
journalctl -u rovy.service -f
```

---

## How It Works

```
Your Voice → Microphone (USB)
    ↓
PyAudio (captures 44.1kHz audio)
    ↓
Resample to 16kHz
    ↓
Stream to Deepgram via WebSocket
    ↓
Real-time Transcription
    ↓
Wake Word Detection (fuzzy matching)
    ↓
Trigger Robot Response
```

---

## Usage in Code

```python
from wake_word_detector_deepgram import DeepgramWakeWordDetector
import asyncio

detector = DeepgramWakeWordDetector(
    api_key="your_key",
    wake_words=["hey rovy"],
    device_sample_rate=44100,
)

async def listen():
    detected = await detector.listen_for_wake_word_async(
        callback=lambda text: print(f"Heard: {text}"),
        timeout=30.0
    )
    detector.cleanup()

asyncio.run(listen())
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: deepgram` | Run `./install_deepgram.sh` |
| `Failed to start connection` | Check API key in config.py |
| `No microphone found` | Run `arecord -l` to list devices |
| Event handlers not working | Use fixed version (handlers without `self`) |

---

## Cost Calculator

- **Free tier**: $200 credit = ~773 hours
- **Continuous use**: ~$6.24/day or ~$187/month
- **Tip**: Only activate when robot is awake

---

## File Reference

```
robot/
├── wake_word_detector_deepgram.py     ← Main corrected implementation
├── install_deepgram.sh                ← Run this first
├── test_deepgram.py                   ← Test installation
├── example_deepgram_wake_word.py      ← Try it out
└── config.py                          ← Add API key here
```

---

## Need Help?

1. **Full Documentation**: See `DEEPGRAM_IMPLEMENTATION.md`
2. **Deepgram Docs**: https://developers.deepgram.com/docs
3. **Get API Key**: https://console.deepgram.com/
4. **Check Logs**: `journalctl -u rovy.service -f`

---

**Status**: ✅ Fixed and Ready to Use

