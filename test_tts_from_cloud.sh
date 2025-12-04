#!/bin/bash
# Test TTS connectivity from cloud to Pi

PI_IP="${ROVY_ROBOT_IP:-100.72.107.106}"
echo "Testing TTS endpoint at: http://${PI_IP}:8000/speak"
echo ""

echo "1. Testing network connectivity..."
if ping -c 2 ${PI_IP} > /dev/null 2>&1; then
    echo "✅ Pi is reachable via ping"
else
    echo "❌ Cannot ping Pi at ${PI_IP}"
    echo "   Check Tailscale connection!"
    exit 1
fi

echo ""
echo "2. Testing HTTP connectivity..."
if curl -s -m 5 "http://${PI_IP}:8000/health" > /dev/null; then
    echo "✅ Pi HTTP service is responding"
else
    echo "❌ Cannot reach Pi HTTP service"
    echo "   Check if rovy service is running on Pi"
    exit 1
fi

echo ""
echo "3. Testing TTS endpoint..."
response=$(curl -s -w "\n%{http_code}" -X POST "http://${PI_IP}:8000/speak" \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello from cloud"}')

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

echo "HTTP Status: ${http_code}"
echo "Response: ${body}"

if [ "$http_code" = "200" ]; then
    echo "✅ TTS endpoint working!"
else
    echo "❌ TTS endpoint returned error ${http_code}"
    if [ "$http_code" = "503" ]; then
        echo "   503 = Service Unavailable"
        echo "   Piper TTS may not be loaded or audio device failed"
    fi
fi

echo ""
echo "4. Environment check on cloud:"
echo "   ROVY_ROBOT_IP = ${ROVY_ROBOT_IP:-<not set, using default>}"

