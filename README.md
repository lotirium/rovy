# Rovy Cloud Server

AI-powered robot assistant server using **100% LOCAL models** - no cloud APIs needed!

## Features

- 🤖 **Local LLM** - Text/chat using Gemma, Llama, or Mistral via llama.cpp
- 👁️ **Local Vision** - Image understanding using LLaVA or Phi-3-Vision
- 🎤 **Local Speech Recognition** - Whisper running locally
- 🔊 **Local TTS** - Natural voice synthesis using Piper
- 👤 **Face Recognition** - Recognize known faces using dlib CNN

## Requirements

- Python 3.9+
- GPU recommended (NVIDIA CUDA for best performance)
- ~8GB RAM minimum (16GB+ recommended)
- ~10GB disk space for models

## Installation

```bash
# Clone or copy this folder
cd rovy_cloud

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration with llama.cpp:
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
```

## Download Models

### 1. Text Model (choose one)
Download GGUF format models from [HuggingFace](https://huggingface.co/):

```bash
# Gemma 2B (small, fast)
wget https://huggingface.co/google/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf

# Or Llama 2 7B
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

### 2. Vision Model (for image understanding)

```bash
# LLaVA v1.5 7B
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf
```

### 3. Piper Voice (for TTS)

```bash
# Download voice model
mkdir -p ~/.local/share/piper-voices
wget -O ~/.local/share/piper-voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
```

### 4. Place Models

Place downloaded models in `~/.cache/` or set environment variables:

```bash
export ROVY_TEXT_MODEL=/path/to/gemma-2-2b-it-Q4_K_M.gguf
export ROVY_VISION_MODEL=/path/to/llava-v1.5-7b-q4.gguf
export ROVY_VISION_MMPROJ=/path/to/mmproj-model-f16.gguf
```

## Usage

### Start the Server

```bash
cd rovy_cloud/server
python main.py
```

The server will:
1. Load local LLM models
2. Initialize Whisper for speech recognition
3. Load Piper for text-to-speech
4. Start WebSocket server on port 8765

### Connect a Client

The server accepts WebSocket connections. Send JSON messages:

```python
import websockets
import asyncio
import json

async def test():
    async with websockets.connect("ws://localhost:8765") as ws:
        # Text query
        await ws.send(json.dumps({
            "type": "text_query",
            "text": "Hello! What can you do?"
        }))
        response = await ws.recv()
        print(response)

asyncio.run(test())
```

### Message Types

**Send to Server:**
- `audio_data` - Raw audio for speech recognition
- `image_data` - Camera frame for vision
- `text_query` - Direct text input
- `sensor_data` - Battery, IMU readings

**Receive from Server:**
- `speak` - Text + audio to play
- `move` - Movement command
- `gimbal` - Camera control
- `display` - OLED text

## Configuration

Edit `shared/config.py` or set environment variables:

```python
# Model paths
ROVY_TEXT_MODEL=/path/to/model.gguf
ROVY_VISION_MODEL=/path/to/vision.gguf
ROVY_VISION_MMPROJ=/path/to/mmproj.gguf

# Server settings
server_config.host = "0.0.0.0"
server_config.port = 8765
server_config.whisper_model = "base"  # tiny/base/small/medium/large
```

## Known Faces

Add face images to `known_faces/` directory:
- Name files as `PersonName.jpg`
- One face per image
- Clear, front-facing photos work best

## Architecture

```
rovy_cloud/
├── server/
│   ├── main.py          # WebSocket server
│   ├── assistant.py     # LLM (local llama.cpp)
│   ├── speech.py        # STT/TTS (Whisper + Piper)
│   └── vision.py        # Face recognition
├── shared/
│   ├── config.py        # Configuration
│   └── messages.py      # Message protocol
├── known_faces/         # Face images
└── requirements.txt
```

## Troubleshooting

### "Model not found"
Set the model path explicitly:
```bash
export ROVY_TEXT_MODEL=/full/path/to/model.gguf
```

### "CUDA out of memory"
Reduce GPU layers:
```python
# In config.py
server_config.n_gpu_layers = 20  # Instead of -1 (all)
```

### "Whisper slow"
Use smaller model:
```python
server_config.whisper_model = "tiny"  # Fastest
```

### "No face recognition"
Install dlib with GPU support:
```bash
pip install dlib --verbose  # Needs CUDA toolkit
```

## Credits

Built with:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Local LLM inference
- [Whisper](https://github.com/openai/whisper) - Speech recognition
- [Piper](https://github.com/rhasspy/piper) - Text-to-speech
- [face_recognition](https://github.com/ageitgey/face_recognition) - Face detection

Ported from original rovy project for Jetson.

