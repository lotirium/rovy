# Robot2 - Raspberry Pi Server for Cloud2

Simple REST API server that runs on Raspberry Pi to provide TTS (text-to-speech) for Cloud2 autonomous android personality.

## Purpose

Cloud2 (autonomous android personality) sends HTTP POST requests to Robot2's `/speak` endpoint to make the android speak through Pi speakers.

## Features

- **REST API**: Simple FastAPI server on port 8000
- **TTS Endpoint**: `/speak` endpoint receives text from Cloud2 and plays it
- **Piper TTS**: Uses Piper for high-quality speech synthesis
- **Fallback**: Falls back to espeak if Piper not available
- **Multi-language**: Supports multiple languages via Piper voices

## Installation

1. **Install dependencies:**
```bash
cd robot2
pip install -r requirements.txt
```

2. **Install Piper TTS (optional but recommended):**
```bash
# Follow instructions at: https://github.com/rhasspy/piper
# Download voice models to: /home/rovy/rovy_client/models/piper/
```

3. **Configure:**
Edit `config.py` or set environment variables:
- `CLOUD2_IP`: IP of PC running Cloud2 (default: 100.121.110.125)
- `ROVY_SERIAL_PORT`: Serial port for rover (default: /dev/ttyAMA0)
- `API_PORT`: API server port (default: 8000)

## Usage

### Manual Start:
```bash
cd robot2
python main.py
```

### Using Start Script:
```bash
cd robot2
./start.sh
```

### Install as Service (auto-start on boot):
```bash
cd robot2
chmod +x install-service.sh
./install-service.sh
```

Then control with:
```bash
sudo systemctl start rovy2    # Start
sudo systemctl stop rovy2      # Stop
sudo systemctl status rovy2   # Status
journalctl -u rovy2 -f        # View logs
```

## API Endpoints

### `POST /speak`
Text-to-speech endpoint for Cloud2.

**Request:**
```json
{
  "text": "Hello, I am an android.",
  "language": "en"
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Speech played",
  "text": "Hello, I am an android.",
  "language": "en"
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "Robot2",
  "tts_available": true
}
```

### `GET /`
Root endpoint with API information.

## How It Works with Cloud2

1. Cloud2 (running on PC) observes through OAK D camera
2. Cloud2 generates thoughts using OpenAI
3. Cloud2 sends HTTP POST to `http://<PI_IP>:8000/speak` with text
4. Robot2 receives request and plays TTS through Pi speakers
5. Android speaks its thoughts

## Configuration

Edit `config.py` or set environment variables:

- `CLOUD2_IP`: Cloud2 server IP (default: 100.121.110.125)
- `ROVY_SERIAL_PORT`: Serial port (default: /dev/ttyAMA0)
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `PIPER_VOICES`: Dictionary of language -> voice model paths

## Notes

- Robot2 is simpler than Robot - it only provides TTS, no WebSocket streaming
- Cloud2 uses HTTP to communicate with Robot2 (not WebSocket)
- Make sure Pi is accessible from Cloud2 PC (same network or Tailscale)
- Piper TTS provides better quality than espeak, but espeak works as fallback

