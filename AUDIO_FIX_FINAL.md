# Audio Error Root Cause & Solutions

## Summary
After extensive debugging, the ALSA audio errors are caused by **USB hardware/driver incompatibility**, not software conflicts.

## What Was Fixed
1. ✅ **Audio mutex lock** - Prevents PyAudio/aplay simultaneous access
2. ✅ **WirePlumber disabled** - No PipeWire interference  
3. ✅ **USB autosuspend disabled** - Power management fixed
4. ✅ **Wake word detection works** - Successfully detects, records, and plays responses

## Remaining Issue
The **Jieli Technology UACDemoV1.0** (Card 2) USB audio device times out after ~60 seconds:
```
⚠️ Audio read timeout - device may be busy or stream corrupted
Expression 'alsa_snd_pcm_mmap_begin' failed in ALSA driver
```

This occurs **inside the ALSA kernel driver** when PyAudio calls `stream.read()`. The device firmware/driver is not reliably providing audio data.

## Solutions (in order of recommendation)

### Option 1: Switch to USB Headphone Set (Card 4) - EASIEST
You have another USB audio device that might be more reliable:
```bash
# Find audio device index for Card 4
python3 -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"

# Update config to use Card 4 instead of Card 2
# Edit robot/config.py or set environment variable
```

### Option 2: Use a Different USB Audio Device - RECOMMENDED
Replace the Jieli Technology device with a more reliable one:
- Blue Snowball iCE USB Microphone
- Audio-Technica ATR2USB
- Any USB device with better Linux/ALSA support

### Option 3: Reduce Buffer Size / Adjust ALSA Settings
Try reducing the audio chunk size to prevent buffer underruns:

Edit `robot/config.py`:
```python
CHUNK_SIZE = 512  # Reduce from 1024
```

Or add ALSA period/buffer configuration in `wake_word_detector_cloud.py`.

### Option 4: Accept Periodic Reinitialization
The current code already handles errors by reinitializing the audio stream. The system recovers automatically, though with brief interruptions.

## Testing the Fix
To test if Card 4 works better, temporarily modify the device detection:
```bash
cd /home/rovy/rovy_client/robot
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f\"Index {i}: {info['name']} - {info['maxInputChannels']} channels\")
"
```

Then update the device index in the code if Card 4 shows better stability.

## Conclusion
The software fixes (mutex, WirePlumber disable, USB power) are working correctly. The remaining issue is **hardware-level USB audio device instability**. Switching to a different USB audio device is the most reliable solution.

