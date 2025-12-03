# ROVY Quick Start Guide

## 🚀 Starting Your Robot

### On Raspberry Pi (Robot):

**Service runs automatically on boot!**

```bash
# Check status
sudo systemctl status rovy.service

# View logs
sudo journalctl -u rovy.service -f

# Restart if needed
sudo systemctl restart rovy.service
```

---

### On PC/Cloud (AI Server):

**Choose one method:**

#### Method 1: Manual Start (Easy)
```bash
cd /home/rovy/rovy_client/cloud
./start_cloud.sh
```
Press Ctrl+C to stop.

#### Method 2: Auto-Start Service
```bash
# Install (once)
cd /home/rovy/rovy_client/cloud/scripts
sudo ./install-service.sh

# Start service
sudo systemctl start rovy-cloud.service

# View logs
sudo journalctl -u rovy-cloud.service -f
```

---

## ✅ Verify It's Working

### On Pi:
```bash
sudo journalctl -u rovy.service -n 20
```
Look for:
- ✅ `ROVY RASPBERRY PI CLIENT`
- ✅ `[Rover] Connected on /dev/ttyAMA0`
- ✅ `[Camera] Ready`
- ✅ `[Server] Connected!`

### On Cloud:
Look for:
- ✅ `WebSocket server running on ws://0.0.0.0:8765`
- ✅ `REST API running on http://0.0.0.0:8000`
- ✅ `🤖 Robot connected: 172.30.1.99:xxxxx`

---

## 📊 Architecture

```
┌─────────────┐
│  Mobile App │
└──────┬──────┘
       │ HTTP :8000
       ↓
┌─────────────────┐
│  Cloud Server   │  ← Run on PC
│  - AI Models    │
│  - REST API     │
│  - WebSocket    │
└──────┬──────────┘
       ↑ WebSocket :8765
       │
┌──────┴──────────┐
│  Robot (Pi)     │  ← Auto-starts on boot
│  - Hardware     │
│  - Camera       │
│  - Stream       │
└─────────────────┘
```

---

## 📝 Useful Commands

### Pi (Robot):
```bash
# Service status
sudo systemctl status rovy.service

# Restart robot
sudo systemctl restart rovy.service

# Live logs
sudo journalctl -u rovy.service -f

# Stop robot
sudo systemctl stop rovy.service

# Disable auto-start
sudo systemctl disable rovy.service
```

### Cloud:
```bash
# Start manually
cd /home/rovy/rovy_client/cloud && ./start_cloud.sh

# Or with service
sudo systemctl start rovy-cloud.service
sudo systemctl status rovy-cloud.service
sudo journalctl -u rovy-cloud.service -f
```

---

## 🔧 Troubleshooting

### Robot won't connect to cloud:
1. Check cloud server is running
2. Check IP in `/home/rovy/rovy_client/robot/config.py`
3. Verify network connectivity (Tailscale)

### Rover hardware not found:
```bash
# Check serial port
ls -la /dev/ttyAMA0

# Add user to dialout group
sudo usermod -a -G dialout rovy
```

### Camera issues:
```bash
# Check camera
ls -la /dev/video*
v4l2-ctl --list-devices
```

---

## 📚 More Information

- **Architecture Details:** See `ARCHITECTURE.md`
- **Cloud Setup:** See `cloud/RUN_CLOUD_SERVER.md`
- **Fixes Applied:** See `ARCHITECTURE_FIXES.md`

---

## 🎯 TL;DR

1. **Pi:** Service runs automatically ✅
2. **Cloud:** Run `./start_cloud.sh` in cloud folder
3. Robot connects automatically when cloud is running! 🚀

