# Quick Start Guide

## Setup Your API Key

Your OpenAI API key is already configured in the start scripts. You can also set it as an environment variable:

### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

### Linux/Mac:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Run the Android

### Windows:
```powershell
cd cloud2
.\start.ps1
```

### Linux/Mac:
```bash
cd cloud2
./start.sh
```

### Or directly:
```bash
cd cloud2
python main.py
```

The android will:
1. Initialize OAK D camera (if available)
2. Connect to OpenAI API
3. Start observing and thinking autonomously
4. Speak through Pi speakers

Press Ctrl+C to stop.

