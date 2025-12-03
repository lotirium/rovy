# Rovy Architecture - Correct Separation

## Overview
The codebase has **two completely separate components** that should NEVER be mixed:

### 1. Robot Client (Raspberry Pi)
**Location:** `/robot/` folder
**Purpose:** Runs ON the Raspberry Pi, connects TO the cloud server
**Entry Point:** `robot/main.py`
**Service:** `robot/rovy.service`
**Startup:** `robot/start.sh`

**What it does:**
- Connects to rover hardware (ESP32) via serial
- Captures video from webcam
- Records audio from ReSpeaker
- **Connects as WebSocket CLIENT to cloud server (port 8765)**
- Streams audio/video TO cloud
- Receives commands FROM cloud (speak, move, etc.)

**Key files:**
- `robot/main.py` - WebSocket client that connects to cloud
- `robot/config.py` - Pi configuration (SERVER_URL points to cloud)
- `robot/rover.py` - Hardware interface for rover
- `robot/wifi_provision.py` - WiFi hotspot setup

---

### 2. Cloud Server (PC/Cloud)
**Location:** `/cloud/` folder
**Purpose:** Runs ON a PC with GPU, processes AI
**Entry Point:** `cloud/main.py` (recommended) OR `cloud/scripts/autorun.sh` (REST API only)
**Service:** `cloud/scripts/api.service`
**Startup:** `cloud/start_cloud.sh`

**What it does:**
- **WebSocket SERVER on port 8765** - Receives connections FROM robot
- **FastAPI REST API on port 8000** - For mobile app
- AI processing: LLM (Qwen2-VL), Vision, STT (Whisper), TTS (Piper)
- Processes audio/video from robot
- Sends commands back to robot

**Key files:**
- `cloud/main.py` - Unified server (REST + WebSocket for robot)
- `cloud/app/main.py` - FastAPI app (REST + WebSocket for mobile)
- `cloud/config.py` - Cloud configuration
- `cloud/ai.py` - AI models (Qwen2-VL)
- `cloud/speech.py` - STT/TTS (Whisper + Piper)

---

## Network Communication

```
┌─────────────────────┐
│  Mobile App         │
│  (React Native)     │
└──────────┬──────────┘
           │
           │ HTTP REST (port 8000)
           │ WebSocket (port 8000)
           │
           v
┌─────────────────────────────────────┐
│  Cloud Server (PC)                  │
│  /cloud/                            │
│                                     │
│  ┌─────────────────────────────┐  │
│  │ FastAPI (port 8000)         │  │
│  │ - REST API for mobile       │  │
│  │ - WebSocket for mobile      │  │
│  └─────────────────────────────┘  │
│                                     │
│  ┌─────────────────────────────┐  │
│  │ WebSocket (port 8765)       │  │
│  │ - Receives robot connection │  │
│  │ - Gets audio/video streams  │  │
│  │ - Sends commands            │  │
│  └─────────────────────────────┘  │
│                                     │
│  ┌─────────────────────────────┐  │
│  │ AI Processing               │  │
│  │ - Qwen2-VL (LLM + Vision)   │  │
│  │ - Whisper (STT)             │  │
│  │ - Piper (TTS)               │  │
│  └─────────────────────────────┘  │
└───────────────┬─────────────────────┘
                │
                │ WebSocket (port 8765)
                │
                v
┌─────────────────────────────────────┐
│  Robot Client (Raspberry Pi)        │
│  /robot/                            │
│                                     │
│  ┌─────────────────────────────┐  │
│  │ WebSocket Client            │  │
│  │ - Connects to cloud:8765    │  │
│  │ - Streams audio/video       │  │
│  │ - Receives commands         │  │
│  └─────────────────────────────┘  │
│                                     │
│  ┌─────────────────────────────┐  │
│  │ Hardware Interfaces         │  │
│  │ - Rover (ESP32) via serial  │  │
│  │ - Webcam (video)            │  │
│  │ - ReSpeaker (audio)         │  │
│  └─────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## Fixed Issues

### ❌ BEFORE (Wrong):
`robot/start.sh` was doing:
```bash
cd ../cloud
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
**Problem:** Pi service was running the CLOUD server code!

### ✅ AFTER (Correct):
`robot/start.sh` now does:
```bash
# Stay in robot directory and run robot client
python main.py
```
**Result:** Pi service runs ONLY robot client code!

---

## How to Run

### On Raspberry Pi:
```bash
cd /home/rovy/rovy_client/robot
sudo systemctl start rovy.service
# OR manually:
./start.sh
```

### On PC/Cloud:
```bash
cd /home/rovy/rovy_client/cloud
# Option 1: Full server (REST + WebSocket for robot)
python main.py
# OR
./start_cloud.sh

# Option 2: REST API only (no robot connection support)
./scripts/autorun.sh
```

---

## Service Files

### Pi Service: `robot/rovy.service`
- **Working Directory:** `/home/rovy/rovy_client/robot`
- **Runs:** `robot/start.sh` → `robot/main.py`
- **User:** `rovy`

### Cloud Service: `cloud/scripts/api.service`
- **Working Directory:** `/home/jetson/rovy/api` (update this!)
- **Runs:** `cloud/scripts/autorun.sh` → `app.main:app` (REST only)
- **User:** `jetson`

**Note:** The cloud service currently only runs REST API. To support robot connections, it should run `cloud/main.py` instead!

---

## Virtual Environments

**CRITICAL:** Each component has its OWN virtual environment to ensure complete separation:

### Robot Virtual Environment
**Location:** `/home/rovy/rovy_client/robot/venv/`
**Setup:**
```bash
cd /home/rovy/rovy_client/robot
./setup.sh
```

### Cloud Virtual Environment
**Location:** `/home/rovy/rovy_client/cloud/.venv/`
**Setup:**
```bash
cd /home/rovy/rovy_client/cloud
./scripts/setup.sh
```

**DO NOT** share virtual environments between robot and cloud!

---

## Requirements

### Robot (`robot/requirements.txt`):
- websockets - WebSocket client
- opencv-python - Camera capture
- pyaudio - Audio recording
- pyserial - Rover communication
- sounddevice, soundfile - Audio playback

### Cloud (`cloud/requirements.txt`):
- fastapi - REST API
- uvicorn - ASGI server
- websockets - WebSocket server (for robot)
- transformers - AI models
- torch - PyTorch for AI
- whisper - Speech-to-text
- piper-tts - Text-to-speech
- pillow - Image processing

---

## Summary

**Key Rule:** 
- **Pi (`robot/`) code should NEVER import from `cloud/`**
- **Cloud (`cloud/`) code should NEVER be run on the Pi**
- Each folder is completely self-contained with its own config, requirements, and services

The architecture is now properly separated! ✅

