#!/bin/bash
# Start Rovy Robot2
# REST API server for Cloud2 autonomous android

cd "$(dirname "$0")"

echo "================================"
echo "  ROVY ROBOT2 STARTUP"
echo "  (For Cloud2 Android)"
echo "================================"

# Activate virtual environment (robot2-specific)
VENV_PATH="/home/rovy/rovy_client/robot2/venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "WARNING: Virtual environment not found at $VENV_PATH"
    echo "Create it with: python3 -m venv /home/rovy/rovy_client/robot2/venv"
    echo "                pip install -r /home/rovy/rovy_client/robot2/requirements.txt"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Show current IP
IP=$(hostname -I | awk '{print $1}')
echo "✓ IP Address: $IP"
echo "✓ API Server: http://$IP:8000"
echo ""

# Start robot2 server
echo "Starting Robot2 API server..."
echo ""
if [ -f "$VENV_PATH/bin/python" ]; then
    "$VENV_PATH/bin/python" main.py
else
    python3 main.py
fi

