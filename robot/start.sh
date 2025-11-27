#!/bin/bash
# Start Rovy Robot
# 1. WiFi Provisioning (if no WiFi) - creates hotspot for phone setup
# 2. Main API server

cd "$(dirname "$0")"

echo "================================"
echo "  ROVY ROBOT STARTUP"
echo "================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Step 1: WiFi Provisioning (Hotspot if no WiFi)
echo ""
echo "[1/2] Checking WiFi connection..."

# Check if connected to WiFi
WIFI_CONNECTED=$(nmcli -t -f TYPE,STATE device status | grep "wifi:connected" || true)

if [ -z "$WIFI_CONNECTED" ]; then
    echo "No WiFi connection detected."
    echo "Starting WiFi provisioning (hotspot mode)..."
    echo ""
    python3 wifi_provision.py
    
    # Re-check after provisioning
    WIFI_CONNECTED=$(nmcli -t -f TYPE,STATE device status | grep "wifi:connected" || true)
    if [ -z "$WIFI_CONNECTED" ]; then
        echo "ERROR: WiFi still not connected after provisioning"
        exit 1
    fi
else
    echo "✓ WiFi already connected"
fi

# Show current IP
IP=$(hostname -I | awk '{print $1}')
echo "✓ IP Address: $IP"
echo ""

# Step 2: Start main API
echo "[2/2] Starting main API server..."
echo ""
cd ../cloud
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
