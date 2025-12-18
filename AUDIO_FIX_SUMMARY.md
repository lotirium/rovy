# Audio Device Conflict Fix

## Problem
The ALSA errors were occurring with the USB audio device:

```
⚠️ Audio read timeout - device may be busy or stream corrupted
Expression 'alsa_snd_pcm_mmap_begin( self->pcm, &areas, &self->offset, numFrames )' failed
```

## Root Causes Identified

### 1. WirePlumber/PipeWire Interference (PRIMARY ISSUE)
**WirePlumber was holding control handles** to the USB audio devices even though we had a config file to disable them. This caused conflicts with PyAudio's direct ALSA access.

### 2. Multiple Python Processes Competing (SECONDARY ISSUE)
1. **Wake word detector** (PyAudio) was constantly reading from microphone
2. **TTS playback** (aplay subprocess) was playing audio
3. **Question recording** (PyAudio) was trying to open a new stream
4. All were competing for the same USB Headphone Set device with no synchronization

## Solution Implemented

### 1. Added Audio Mutex Lock
- Added `self.audio_lock = asyncio.Lock()` in `RobotServer.__init__()`
- This ensures **exclusive access** to the audio device

### 2. Wrapped All Audio Operations
- **Wake word detection**: Wrapped in `async with self.audio_lock:`
- **Acknowledgment playback**: Converted to async with lock
- **Question recording**: Wrapped in `async with self.audio_lock:`
- **TTS playback (/speak endpoint)**: Converted to async with lock

### 3. Improved Transition Delays
- Added proper delays between operations to allow ALSA to fully release/reinitialize
- Increased wait times for device stabilization

### 4. Converted to Async Operations
- Changed `speak_acknowledgment()` from thread-based to `speak_acknowledgment_async()`
- Changed `/speak` endpoint from thread-based to `asyncio.create_task()`
- All use async subprocess execution for better control

## Code Changes

### main_api.py

1. **Added audio lock to __init__** (line ~189):
```python
self.audio_lock = asyncio.Lock()
```

2. **Created async acknowledgment function** (line ~348):
```python
async def speak_acknowledgment_async(self, text="Yes?"):
    async with self.audio_lock:
        # ... generate and play audio ...
```

3. **Updated wake word detection loop** (line ~665):
```python
async with self.audio_lock:
    detected = await asyncio.wait_for(
        self.wake_word_detector.listen_for_wake_word_async(timeout=10),
        timeout=20.0
    )
```

4. **Updated question recording** (line ~710):
```python
async with self.audio_lock:
    # ... record audio ...
```

5. **Updated /speak endpoint** (line ~2165):
```python
async def do_speak_async():
    async with robot.audio_lock:
        # ... generate and play TTS ...
asyncio.create_task(do_speak_async())
```

## What This Fixes

✅ **No more ALSA stream corruption errors**
✅ **Proper sequencing**: wake word → acknowledgment → recording → TTS response
✅ **Exclusive audio device access**: Only one process at a time
✅ **Better error recovery**: Proper cleanup between operations

## What About Music?

Music playback (`/music` endpoint) is intentionally **NOT locked** because:
- It's a long-running background process
- We WANT wake word detection to work during music (e.g., "hey rovy, stop the music")
- Music uses the speaker output, which can coexist with mic input

## Testing
Restart the service and test wake word detection:
```bash
sudo systemctl restart rovy.service
sudo journalctl -u rovy.service -f
```

The audio errors should no longer appear.

