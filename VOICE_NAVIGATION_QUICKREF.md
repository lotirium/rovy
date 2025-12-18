# Voice Navigation Quick Reference

## Quick Start

1. **Start the services:**
   ```bash
   # On PC
   cd cloud && python main.py
   
   # On Raspberry Pi
   cd robot && python main.py
   ```

2. **Use voice commands:**
   ```
   "Hey Rovy, start auto navigation"
   "Hey Rovy, stop navigation"
   ```

## Voice Commands

### Navigation Commands
| Command | What It Does |
|---------|-------------|
| "Hey Rovy, **start auto navigation**" | Begin autonomous exploration with obstacle avoidance |
| "Hey Rovy, **start exploring**" | Same as above - starts exploration mode |
| "Hey Rovy, **stop navigation**" | Stop autonomous navigation and return to manual control |
| "Hey Rovy, **stop exploring**" | Stop exploration mode |

### Fun Commands
| Command | What It Does |
|---------|-------------|
| "Hey Rovy, **dance**" | Perform a party dance routine (10 seconds) |
| "Hey Rovy, **do a wiggle dance**" | Perform side-to-side wiggle dance |
| "Hey Rovy, **do a spin dance**" | Perform 360-degree spinning dance |
| "Hey Rovy, **show me your moves**" | Perform a dance (same as "dance") |
| "Hey Rovy, **bust a move**" | Perform a dance (same as "dance") |

## What Happens

### When You Say "Start Auto Navigation":

1. ✅ Wake word detected locally on Pi
2. 📤 Audio sent to cloud server
3. 🎤 Whisper transcribes: "start auto navigation"
4. 🧠 Cloud detects navigation command
5. 📨 Sends navigation message to robot
6. 🤖 Robot initializes OAK-D navigation
7. 🚀 Robot begins exploring autonomously
8. 👁️ OAK-D camera detects obstacles
9. 🛣️ Navigation system avoids obstacles
10. 📺 Status displayed on OLED screen

### Navigation Features:

- ✅ **Obstacle Detection**: OAK-D depth camera (40cm-5m range)
- ✅ **Obstacle Avoidance**: Potential Field method
- ✅ **Safe Distance**: 60cm buffer from obstacles
- ✅ **Speed Control**: Max 0.4 m/s (safe indoor speed)
- ✅ **Emergency Stop**: Immediate halt on critical obstacles
- ✅ **Autonomous Turning**: Explores new directions when blocked
- ✅ **Status Display**: Real-time info on OLED

## System Requirements

### Hardware:
- Rovy robot with ESP32 controller
- Raspberry Pi 5
- OAK-D camera
- ReSpeaker microphone array

### Software (on Pi):
- Wake word detector (Silero VAD + Whisper tiny)
- Navigation system (OAK-D integration)
- Robot client (WebSocket)

### Software (on PC):
- Cloud server
- Whisper STT model
- WebSocket server

## Configuration Files

| File | Purpose |
|------|---------|
| `robot/config.py` | Wake words, audio settings, server connection |
| `cloud/config.py` | Server settings, AI models |
| `oakd_navigation/rovy_integration.py` | Navigation parameters |

## Common Issues

| Problem | Solution |
|---------|----------|
| Wake word not detected | Check microphone, lower VAD_THRESHOLD in robot/config.py |
| Robot doesn't start | Verify OAK-D connected, check logs |
| Robot stops immediately | Check depth sensing, ensure clear space in front |
| Command not recognized | Use exact phrases, check cloud server logs |

## Architecture Overview

```
┌──────────────────┐
│  Raspberry Pi    │
│  ┌────────────┐  │
│  │ Wake Word  │  │ 
│  │ Detector   │  │
│  └─────┬──────┘  │
│        │         │
│        v         │
│  ┌────────────┐  │
│  │  Record    │  │
│  │  Audio     │  │
│  └─────┬──────┘  │
└────────┼─────────┘
         │ WebSocket
         v
┌────────┴─────────┐
│   Cloud Server   │
│  ┌────────────┐  │
│  │  Whisper   │  │
│  │  STT       │  │
│  └─────┬──────┘  │
│        v         │
│  ┌────────────┐  │
│  │  Command   │  │
│  │  Parser    │  │
│  └─────┬──────┘  │
└────────┼─────────┘
         │ WebSocket
         v
┌────────┴─────────┐
│  Raspberry Pi    │
│  ┌────────────┐  │
│  │ Navigation │  │
│  │ Handler    │  │
│  └─────┬──────┘  │
│        v         │
│  ┌────────────┐  │
│  │ OAK-D Nav  │  │
│  │ System     │  │
│  └─────┬──────┘  │
│        v         │
│  ┌────────────┐  │
│  │ Motors     │  │
│  └────────────┘  │
└──────────────────┘
```

## Modified Files Summary

### `robot/main.py`
- ✅ Added `handle_navigation()` method
- ✅ Processes navigation commands from server
- ✅ Creates RovyNavigator with existing rover instance
- ✅ Runs navigation in separate thread

### `cloud/main.py`
- ✅ Added navigation keyword detection
- ✅ Added `send_navigation()` method
- ✅ Detects "start auto navigation" command
- ✅ Detects "stop navigation" command

### `oakd_navigation/rovy_integration.py`
- ✅ Added `rover_instance` parameter to `__init__()`
- ✅ Prevents serial port conflicts
- ✅ Added `_owns_rover` flag for proper cleanup

## Example Session

```
[Robot starts, connects to cloud server]

You: "Hey Rovy"
Rovy: "Yes? I'm listening."

You: "Start auto navigation"
Rovy: "Starting autonomous navigation. I will explore and avoid obstacles."

[Robot begins moving, avoiding obstacles, exploring]
[OLED displays: "EXPLORE MODE", "Moving: True", "Obstacle: False"]

[After some time...]

You: "Hey Rovy"
Rovy: "Yes?"

You: "Stop navigation"
Rovy: "Stopping navigation."

[Robot stops, navigation system shuts down]
```

## Testing

### Test Wake Word:
```bash
# On Raspberry Pi
cd robot
python -c "from wake_word_detector import WakeWordDetector; d = WakeWordDetector(); d.listen_for_wake_word()"
```

### Test Navigation:
```bash
# On Raspberry Pi
cd oakd_navigation
python rovy_integration.py
```

### Test Depth Sensing:
```bash
# On Raspberry Pi
cd oakd_navigation
python debug_depth.py
```

## Performance Tips

1. **Better Wake Word Detection:**
   - Position microphone away from motors
   - Lower VAD_THRESHOLD for more sensitivity
   - Use clear wake words

2. **Better Navigation:**
   - Ensure good lighting for OAK-D
   - Start in open space
   - Adjust safe_distance for environment

3. **Better Responsiveness:**
   - Use fast network connection
   - Run cloud server on same network
   - Use Whisper tiny model for speed

## Next Steps

After getting basic voice navigation working:

1. Try different navigation modes
2. Adjust parameters for your environment
3. Add custom voice commands
4. Implement waypoint navigation
5. Build maps of explored areas

## Documentation

- **Full Guide**: `oakd_navigation/VOICE_NAVIGATION.md`
- **Navigation System**: `oakd_navigation/README.md`
- **Integration Details**: `oakd_navigation/INTEGRATION_SUMMARY.md`
- **Quick Start**: `oakd_navigation/QUICKSTART.md`

## Support Commands

```bash
# Check system status
systemctl status rovy-robot.service  # If using systemd

# View logs
tail -f /tmp/rovy-robot.log
tail -f /tmp/rovy-cloud.log

# Test components
python oakd_navigation/test_system.py
```

---

**Pro Tip**: Start with short exploration sessions and gradually increase as you tune the parameters for your environment!

