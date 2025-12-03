# ROVY Final Architecture - Fully Separated ✅

## Overview

The ROVY system now has **complete separation** between robot and cloud, with proper distributed architecture.

---

## Architecture Diagram

```
┌─────────────────────┐
│   Mobile App        │
│   (React Native)    │
└──────────┬──────────┘
           │
           │ REST API (port 8000)
           │ Camera, Control, Status
           │
           v
┌─────────────────────────────────────┐
│  Raspberry Pi (Robot Server)        │
│  /robot/main_api.py                 │
│                                     │
│  ✅ REST API Server (port 8000)     │
│     - Camera streams (/video, /shot) │
│     - Robot control (/control/*)     │
│     - Status endpoints               │
│     - WebSocket (/json, /camera/ws)  │
│                                     │
│  ✅ Hardware Control                 │
│     - Rover (ESP32) via serial      │
│     - Camera (USB/CSI)              │
│     - Audio (ReSpeaker)             │
│     - OLED Display                  │
│                                     │
│  ✅ Cloud Streaming Client           │
│     - Streams to PC for AI          │
└─────────────┬───────────────────────┘
              │
              │ WebSocket Client
              │ Port 8765
              │ Streams: video, audio, sensors
              │
              v
┌─────────────────────────────────────┐
│  PC/Cloud Server                    │
│  /cloud/main.py                     │
│                                     │
│  ✅ WebSocket Server (port 8765)     │
│     - Receives from robot           │
│     - Processes with AI             │
│                                     │
│  ✅ REST API (port 8000)             │
│     - AI chat endpoint              │
│     - Optional mobile access        │
│                                     │
│  ✅ AI Processing                    │
│     - Qwen2-VL (LLM + Vision)       │
│     - Whisper (Speech-to-Text)      │
│     - Piper (Text-to-Speech)        │
└─────────────────────────────────────┘
```

---

## Component Details

### 🤖 Raspberry Pi - Robot Server

**File:** `/robot/main_api.py`  
**Port:** 8000 (REST API)  
**Purpose:** Direct hardware access for mobile app + streaming to cloud

#### Features:
- **FastAPI REST Server**
  - Camera streaming (MJPEG, WebSocket, snapshots)
  - Robot control (move, stop, lights, gimbal)
  - Status monitoring (battery, sensors)
  - WiFi management
  
- **Hardware Interfaces**
  - Rover control via serial (`/dev/ttyAMA0`)
  - Camera capture (OpenCV)
  - Audio recording (PyAudio)
  - OLED display
  
- **Cloud Streaming**
  - WebSocket client connects to PC
  - Streams video for AI vision processing
  - Streams audio for speech recognition
  - Sends sensor data

#### Key Dependencies:
```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
websockets>=11.0
opencv-python>=4.5.0
pyserial>=3.5
```

---

### ☁️ PC/Cloud - AI Server

**File:** `/cloud/main.py`  
**Ports:** 8000 (REST API), 8765 (WebSocket)  
**Purpose:** AI processing and mobile app AI features

#### Features:
- **WebSocket Server (port 8765)**
  - Receives video stream from robot
  - Receives audio stream from robot
  - Processes with AI models
  - Sends back AI responses
  
- **REST API (port 8000)**
  - Chat endpoint (`/chat`)
  - Vision endpoint (`/vision`)
  - Speech-to-text (`/stt`)
  - Text-to-speech (`/tts`)
  
- **AI Models**
  - Qwen2-VL: Large language model + vision
  - Whisper: Speech-to-text
  - Piper: Text-to-speech

#### Key Dependencies:
```
fastapi
uvicorn
websockets
transformers
torch
whisper
piper-tts
```

---

### 📱 Mobile App

**Purpose:** User interface for robot control

#### Connection Points:
1. **Robot REST API** (`http://pi-ip:8000`)
   - Camera streams
   - Robot control
   - Status monitoring
   
2. **Cloud REST API** (`http://pc-ip:8000`) - Optional
   - AI chat
   - Advanced vision processing

---

## Data Flow

### 1. Mobile App → Robot Control
```
Mobile App
    ↓ POST /control/move
Pi Robot Server
    ↓ Serial command
ESP32 Rover
    → Motors move
```

### 2. Mobile App → Camera View
```
Mobile App
    ↓ GET /video
Pi Robot Server
    ↓ OpenCV capture
USB Camera
    → MJPEG stream
```

### 3. Robot → Cloud AI Processing
```
Pi Robot Server
    ↓ WebSocket (port 8765)
PC Cloud Server
    ↓ Qwen2-VL model
AI Processing
    ↓ WebSocket response
Pi Robot Server
    → Execute action
```

---

## Folder Structure

### Raspberry Pi (`/robot/`)
```
robot/
├── main_api.py          ← Main server (REST + Cloud client)
├── main.py              ← Old client-only version (deprecated)
├── rover.py             ← Rover hardware interface
├── config.py            ← Robot configuration
├── wifi_provision.py    ← WiFi setup
├── requirements.txt     ← Python dependencies
├── venv/                ← Isolated virtual environment
├── start.sh             ← Startup script
├── setup.sh             ← Environment setup
├── rovy.service         ← Systemd service
└── install-service.sh   ← Service installer
```

### PC/Cloud (`/cloud/`)
```
cloud/
├── main.py              ← Unified cloud server (WebSocket + REST)
├── ai.py                ← AI models (Qwen2-VL)
├── speech.py            ← STT/TTS (Whisper, Piper)
├── config.py            ← Cloud configuration
├── app/
│   └── main.py          ← FastAPI REST API
├── requirements.txt     ← Python dependencies
├── .venv/               ← Isolated virtual environment
├── start_cloud.sh       ← Startup script
└── scripts/
    ├── setup.sh         ← Environment setup
    └── install-service.sh ← Service installer
```

---

## Key Benefits ✅

1. **Complete Separation**
   - Robot and cloud are independent
   - Each has own folder, venv, dependencies
   - No cross-imports

2. **Mobile App Works Locally**
   - Direct hardware access on Pi
   - No cloud required for basic operation
   - Fast response times

3. **Cloud AI Enhancement**
   - Powerful GPU processing on PC
   - Advanced AI capabilities
   - Optional - system works without it

4. **Clean Architecture**
   - Clear responsibilities
   - Easy to deploy
   - Easy to maintain

5. **Can Delete Cloud from Pi**
   - Robot folder is self-contained
   - No dependencies on cloud code
   - True separation achieved ✅

---

## Running the System

### On Raspberry Pi:
```bash
# Auto-starts on boot via systemd
sudo systemctl status rovy.service

# Or manually:
cd /home/rovy/rovy_client/robot
./start.sh
```

**Mobile app connects to:** `http://<pi-ip>:8000`

### On PC/Cloud (Optional):
```bash
cd /home/rovy/rovy_client/cloud
./start_cloud.sh
```

**Robot streams to:** `ws://<pc-ip>:8765`

---

## Verification

### Check Robot is Running:
```bash
sudo systemctl status rovy.service
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "ok",
  "version": "2.0",
  "capabilities": {
    "camera": true,
    "rover": true,
    "audio": true,
    "cloud_stream": true
  }
}
```

### Check Cloud is Running:
```bash
# On PC
curl http://localhost:8000/health
curl http://localhost:8765/
```

---

## Git Commits

The architecture was fixed in these commits:

1. **7b1ed4b** - Fixed architecture separation (robot runs robot code, not cloud)
2. **8bc8caf** - Fixed serial port configuration
3. **30212aa** - Added cloud service documentation
4. **d01d22d** - Added REST API to robot server ✅
5. **f14b620** - Added FastAPI dependencies

---

## Summary

**Before:** Mixed architecture, Pi ran cloud server  
**After:** Clean separation, Pi has own API, cloud is optional

✅ **Robot (Pi):** REST API for mobile + hardware control + optional cloud streaming  
✅ **Cloud (PC):** AI processing only  
✅ **Mobile App:** Direct access to robot hardware  
✅ **Architecture:** Fully distributed and properly separated

🎉 **Mission Accomplished!**

