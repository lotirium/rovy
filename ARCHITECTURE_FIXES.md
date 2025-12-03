# Architecture Fixes Applied

## Problem
The robot systemd service was incorrectly running cloud server code, causing the architecture to be merged instead of separated.

---

## Issues Found and Fixed

### 1. ❌ Robot Service Running Cloud Code
**File:** `robot/start.sh`

**BEFORE (Wrong):**
```bash
# Step 2: Start main API
echo "[2/2] Starting main API server..."
echo ""
cd ../cloud
/home/rovy/rovy_client/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Problem:** 
- Robot startup script changed to cloud directory
- Started FastAPI/Uvicorn server (cloud code)
- Made Pi run the cloud server instead of robot client

**AFTER (Fixed):**
```bash
# Step 2: Start robot client
echo "[2/2] Starting robot client..."
echo ""
# Stay in robot directory and run robot client
python main.py
```

**Result:** ✅ Pi now runs only robot client code

---

### 2. ❌ Shared Virtual Environment
**File:** `robot/start.sh`

**BEFORE (Wrong):**
```bash
VENV_PATH="/home/rovy/rovy_client/venv"  # Shared with cloud!
```

**Problem:**
- Both robot and cloud used the same venv at `/home/rovy/rovy_client/venv`
- Dependencies were mixed
- Changes to one could break the other

**AFTER (Fixed):**
```bash
VENV_PATH="/home/rovy/rovy_client/robot/venv"  # Robot-specific
```

**Result:** ✅ Each component has its own isolated virtual environment

**New Setup Script:** Created `robot/setup.sh` to easily create robot's venv

---

## Current Architecture (Correct)

### Robot Client (Raspberry Pi)
```
/home/rovy/rovy_client/robot/
├── main.py              # WebSocket client (connects TO cloud)
├── config.py            # Pi configuration
├── rover.py             # Hardware interface
├── wifi_provision.py    # WiFi setup
├── requirements.txt     # Robot dependencies
├── venv/                # Robot virtual environment
├── setup.sh             # Setup script (NEW)
├── start.sh             # Startup script (FIXED)
└── rovy.service         # Systemd service
```

**What it does:**
- Runs ON Raspberry Pi
- Connects TO cloud WebSocket server (port 8765)
- Streams audio/video to cloud
- Receives commands from cloud

### Cloud Server (PC)
```
/home/rovy/rovy_client/cloud/
├── main.py              # Unified server (WebSocket + REST API)
├── app/
│   └── main.py          # FastAPI REST API
├── ai.py                # AI models (Qwen2-VL)
├── speech.py            # STT/TTS (Whisper + Piper)
├── config.py            # Cloud configuration
├── requirements.txt     # Cloud dependencies
├── .venv/               # Cloud virtual environment
└── scripts/
    ├── autorun.sh       # Systemd startup
    └── setup.sh         # Setup script
```

**What it does:**
- Runs ON PC/Cloud (with GPU)
- WebSocket SERVER on port 8765 (receives FROM robot)
- REST API SERVER on port 8000 (for mobile app)
- AI processing (LLM, Vision, STT, TTS)

---

## Network Flow (Correct)

```
Mobile App
    ↓ (HTTP REST + WebSocket)
    ↓ Port 8000
    ↓
Cloud Server (PC)
    ↓ (WebSocket Server)
    ↓ Port 8765
    ↓
Robot Client (Pi)
    ↓ (Serial)
    ↓
Rover Hardware (ESP32)
```

---

## Verification

### Check Robot Service
```bash
# View service status
sudo systemctl status rovy.service

# Check logs (should say "ROVY ROBOT" not "Cloud Server")
journalctl -u rovy.service -f
```

### Check Cloud Service
```bash
# On PC, run cloud server
cd /home/rovy/rovy_client/cloud
python main.py

# Should show:
# - REST API running on port 8000
# - WebSocket server on port 8765
```

### Verify No Cross-Imports
```bash
# Should find nothing
grep -r "from cloud" robot/
grep -r "import cloud" robot/

# Should find nothing
grep -r "from robot" cloud/
grep -r "import robot" cloud/
```

---

## Summary

✅ **Robot service** now runs ONLY robot client code (`robot/main.py`)  
✅ **Cloud service** runs ONLY cloud server code (`cloud/main.py`)  
✅ Each has its own **separate virtual environment**  
✅ No cross-imports between folders  
✅ Clear separation of concerns  

The architecture is now properly separated! 🎉

---

## Next Steps

1. **On Raspberry Pi:**
   ```bash
   cd /home/rovy/rovy_client/robot
   ./setup.sh  # Create robot venv and install dependencies
   sudo systemctl restart rovy.service
   ```

2. **On PC/Cloud:**
   ```bash
   cd /home/rovy/rovy_client/cloud
   ./scripts/setup.sh  # Ensure cloud venv is set up
   python main.py  # Start cloud server
   ```

3. **Verify Connection:**
   - Robot should connect to cloud WebSocket on port 8765
   - Check robot logs: `journalctl -u rovy.service -f`
   - Check cloud logs in the terminal running `cloud/main.py`

