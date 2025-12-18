# 🕺 Rovy Dancing Mode Integration with Music 🎵

## Overview

Rovy now has a **Dancing Mode with Music** that can be triggered from:
1. **Phone App** - Direct REST API call or voice command through the app
2. **Rover's Microphone** - Voice commands detected locally on the robot

### NEW: Music Integration ✨
- Dances **with music** from YouTube Music
- **Random song selection** - never the same song twice!
- **Genre support** - dance to classical, jazz, rock, EDM, and more
- **Automatic synchronization** - music starts with dance, stops when done

## Dance Styles

The robot supports three dance styles:

### 1. 🎉 Party Dance (Default)
- Energetic spinning left and right
- Flashing front and back lights
- Head shaking movements
- Forward wiggle movements
- **Best for:** General entertainment, parties

### 2. 🐍 Wiggle Dance
- Side-to-side wiggling motion
- Head turning left and right
- Alternating light patterns
- Nodding head movements
- **Best for:** Playful, snake-like movements

### 3. 🌀 Spin Dance
- Continuous 360-degree spins
- Full brightness light pulses
- Head tilting during spins
- Alternating clockwise/counter-clockwise
- **Best for:** Dramatic performances

## How to Trigger Dancing

### Method 1: Voice Command from Rover's Microphone

Simply say one of these commands when near the robot:

```
# Dance with music (NEW! 🎵):
"Hey Rovy, dance!"                      → Dances with dance music
"Hey Rovy, show me your moves!"         → Dances with music
"Hey Rovy, bust a move!"                → Dances with music

# Specify dance style:
"Hey Rovy, do a wiggle dance!"          → Wiggle dance with music
"Hey Rovy, do a spin dance!"            → Spin dance with music
"Hey Rovy, do a party dance!"           → Party dance with music

# Specify music genre:
"Hey Rovy, dance to classical!"         → Dances with classical music
"Hey Rovy, dance to jazz!"              → Dances with jazz music
"Hey Rovy, dance to rock!"              → Dances with rock music
"Hey Rovy, spin dance with electronic!" → Spin dance with electronic music
```

**How it works:**
1. Wake word detected on Pi → Records audio
2. Audio sent to cloud server → Whisper transcribes
3. Cloud detects "dance" keyword → Sends dance command to robot
4. Robot executes dance routine → Displays "DANCE MODE" on OLED

### Method 2: Voice Command from Phone App

Open the Rovy mobile app and use the voice feature:

1. Tap the microphone button
2. Say: "Dance" or "Show me your dance moves"
3. The app will transcribe and send the command
4. Robot starts dancing

### Method 3: REST API from Phone App

The mobile app can directly call the REST endpoint:

```typescript
// Example mobile app code
async function triggerDance(style: 'party' | 'wiggle' | 'spin' = 'party', duration: number = 10) {
  const response = await fetch(`http://${robotIP}:8000/dance`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ style, duration })
  });
  return response.json();
}

// Usage:
await triggerDance('party', 15);  // Party dance for 15 seconds
await triggerDance('wiggle', 8);   // Wiggle dance for 8 seconds
```

### Method 4: Direct HTTP Call

You can trigger dancing from any device on the network:

```bash
# Default party dance for 10 seconds
curl -X POST http://100.72.107.106:8000/dance \
  -H "Content-Type: application/json" \
  -d '{"style": "party", "duration": 10}'

# Wiggle dance for 5 seconds
curl -X POST http://100.72.107.106:8000/dance \
  -H "Content-Type: application/json" \
  -d '{"style": "wiggle", "duration": 5}'

# Spin dance for 15 seconds
curl -X POST http://100.72.107.106:8000/dance \
  -H "Content-Type: application/json" \
  -d '{"style": "spin", "duration": 15}'
```

## API Reference

### REST Endpoint: `/dance`

**POST** `http://<robot-ip>:8000/dance`

**Request Body:**
```json
{
  "style": "party",        // "party", "wiggle", or "spin"
  "duration": 10,          // Duration in seconds (1-60)
  "with_music": true,      // NEW! Play music during dance
  "music_genre": "dance"   // NEW! Music genre (see genres below)
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Dancing party style for 10 seconds with music",
  "style": "party",
  "duration": 10,
  "with_music": true,
  "music_genre": "dance"
}
```

### NEW: REST Endpoint: `/music` 🎵

**POST** `http://<robot-ip>:8000/music`

**Play Music:**
```json
{
  "action": "play",
  "genre": "dance"    // Options: dance, party, classical, jazz, rock, pop, chill, electronic, fun
}
```

**Stop Music:**
```json
{
  "action": "stop"
}
```

**Get Status:**
```json
{
  "action": "status"
}
```

**Error Responses:**
- `400` - Invalid style or duration
- `503` - Rover controller not available
- `501` - Dance function not supported
- `500` - Dance execution failed

### WebSocket Message (Robot Client)

The robot client (running on Pi) receives dance commands via WebSocket:

```json
{
  "type": "dance",
  "style": "party",
  "duration": 10
}
```

## Implementation Details

### Files Modified

1. **`robot/rover.py`**
   - Added `dance(style, duration)` method
   - Implements three dance styles with motor movements, lights, and gimbal

2. **`robot/main.py`**
   - Added `handle_dance(msg)` method
   - Handles WebSocket dance commands from cloud server
   - Displays "DANCE MODE" on OLED during dance

3. **`cloud/main.py`**
   - Added dance keyword detection in voice processing
   - Added `send_dance(websocket, style, duration)` method
   - Detects: "dance", "bust a move", "show me your moves"

4. **`cloud/app/main.py`**
   - Added `/dance` REST endpoint for phone app
   - Added dance detection in voice WebSocket handler
   - Triggers dance via base_controller instance

### Dance Routine Breakdown

Each dance style is a loop that runs for the specified duration:

**Party Dance:**
```python
# Each iteration (~1.4s):
- Flash front lights (255 brightness)
- Spin right (0.3s)
- Flash back lights (255 brightness)
- Spin left (0.3s)
- Shake head left/right
- Wiggle forward with alternating wheel speeds
```

**Wiggle Dance:**
```python
# Each iteration (~1.2s):
- Turn left + tilt head left + front lights (0.4s)
- Turn right + tilt head right + back lights (0.4s)
- Nod head up and down (0.4s)
```

**Spin Dance:**
```python
# Each iteration (~2.4s):
- All lights on
- Full speed clockwise spin (1.0s)
- Lights off briefly
- Full speed counter-clockwise spin (1.0s)
- Tilt head up during spins
```

### Safety Features

- **Maximum Duration:** 60 seconds to prevent overheating
- **Automatic Stop:** Robot stops motors at end of dance
- **Gimbal Reset:** Head returns to center position
- **Lights Off:** LEDs turn off after dance
- **Display Reset:** OLED returns to "Ready" state

## Testing

### Test on Robot (Direct)

SSH into the Raspberry Pi and test the dance function:

```python
from robot.rover import Rover
import time

rover = Rover('/dev/ttyAMA0')
time.sleep(2)

# Test each style
print("Testing party dance...")
rover.dance('party', 5)

print("Testing wiggle dance...")
rover.dance('wiggle', 5)

print("Testing spin dance...")
rover.dance('spin', 5)

rover.cleanup()
```

### Test via REST API

```bash
# From any computer on the network
curl -X POST http://100.72.107.106:8000/dance \
  -H "Content-Type: application/json" \
  -d '{"style": "party", "duration": 5}'
```

### Test via Voice (Robot Microphone)

1. Ensure wake word detector is running: `cd robot && python main.py`
2. Say: "Hey Rovy, dance!"
3. Robot should start dancing with "DANCE MODE" on display

### Test via Voice (Phone App)

1. Open mobile app
2. Tap microphone button
3. Say: "Dance"
4. Robot should start dancing

## Troubleshooting

### Robot doesn't dance when voice command is given

**Check:**
- Is the robot client running? (`python robot/main.py`)
- Is the cloud server running? (`python cloud/main.py`)
- Check cloud server logs for "💃 Dance command detected"
- Check robot logs for "[Dance] Starting"

### Dance command detected but robot doesn't move

**Check:**
- Rover serial connection: Look for "[Rover] Connected on /dev/ttyAMA0"
- Battery level: Dance requires sufficient power
- ESP32 connection: Check if rover controller responds to other commands

### Dance is jerky or incomplete

**Possible causes:**
- Low battery (check voltage)
- Network latency (if triggered remotely)
- CPU overload (check other running processes)

### Phone app can't trigger dance

**Check:**
- Phone is on same network as robot
- Robot IP is correct in app configuration
- REST API is accessible: `curl http://<robot-ip>:8000/health`

## Customizing Dances

You can add your own dance styles by editing `robot/rover.py`:

```python
def dance(self, style='party', duration=10):
    # ... existing code ...
    
    elif style == 'custom':
        # Your custom dance routine
        while time.time() - start_time < duration:
            # Your movements here
            self._send_direct(left_speed, right_speed)
            self.gimbal_ctrl(pan, tilt, 0, 0)
            self.lights_ctrl(front, back)
            time.sleep(delay)
```

Then update the valid styles list in `cloud/app/main.py`:

```python
valid_styles = ['party', 'wiggle', 'spin', 'custom']
```

## Performance Notes

- **CPU Usage:** Minimal (dance runs in separate thread)
- **Battery Impact:** Moderate (motors + LEDs at high duty cycle)
- **Network:** No network needed once command received
- **Memory:** ~5MB for dance routine thread

## Future Enhancements

Potential improvements:
- [ ] Add music synchronization
- [ ] Choreographed multi-robot dances
- [ ] Custom dance sequences from phone app
- [ ] Record and replay custom dances
- [ ] Dance in formation with other robots
- [ ] Add sound effects during dance

## Examples

### Example 1: Quick Party Dance
```bash
curl -X POST http://100.72.107.106:8000/dance \
  -d '{"style": "party", "duration": 5}'
```

### Example 2: Long Spin Performance
```bash
curl -X POST http://100.72.107.106:8000/dance \
  -d '{"style": "spin", "duration": 20}'
```

### Example 3: Voice-Triggered Wiggle
```
User: "Hey Rovy, do a wiggle dance"
Rovy: "Let me show you my wiggle dance moves!" [starts wiggling]
```

### Example 4: Phone App Integration
```typescript
// In your mobile app
const danceButton = () => {
  const styles = ['party', 'wiggle', 'spin'];
  const randomStyle = styles[Math.floor(Math.random() * styles.length)];
  
  fetch(`http://${ROBOT_IP}:8000/dance`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      style: randomStyle, 
      duration: 10 
    })
  });
};
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     DANCING MODE FLOW                        │
└─────────────────────────────────────────────────────────────┘

METHOD 1: Voice from Rover Microphone
┌────────────┐      ┌──────────────┐      ┌──────────────┐
│   User     │─────▶│ Wake Word    │─────▶│ Cloud Server │
│   Voice    │      │ Detector (Pi)│      │  (Whisper)   │
└────────────┘      └──────────────┘      └──────┬───────┘
                                                  │
                                        Detects "dance"
                                                  │
                                                  ▼
                                      ┌──────────────────┐
                                      │ send_dance()     │
                                      │ WebSocket MSG    │
                                      └────────┬─────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Robot Client (Pi)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ handle_dance(msg)                                    │   │
│  │   ├─ Display "DANCE MODE"                           │   │
│  │   └─ Call rover.dance(style, duration)              │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Rover.dance()                                        │   │
│  │   ├─ Motor movements (spin, wiggle, forward)        │   │
│  │   ├─ Gimbal head shaking                            │   │
│  │   └─ LED light flashing                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

METHOD 2: REST API from Phone
┌────────────┐      ┌──────────────────┐      ┌─────────────┐
│ Mobile App │─────▶│ POST /dance      │─────▶│ Rover.dance()│
│  (Button)  │      │ (Pi REST API)    │      │  (Direct)    │
└────────────┘      └──────────────────┘      └─────────────┘

METHOD 3: Voice from Phone App
┌────────────┐      ┌──────────────────┐      ┌─────────────┐
│ Phone Mic  │─────▶│ /voice WebSocket │─────▶│ Pi /dance   │
│   Voice    │      │ Whisper+Detect   │      │  endpoint   │
└────────────┘      └──────────────────┘      └─────────────┘
```

---

**Created:** December 2024  
**Status:** ✅ Fully Integrated  
**Tested:** On Raspberry Pi 5 with ESP32 Rover

