#!/bin/bash
# Verify Rovy Architecture Separation
# Run this to check if robot and cloud are properly separated

echo "================================"
echo "  ARCHITECTURE VERIFICATION"
echo "================================"
echo ""

ERRORS=0

# Check 1: No cross-imports
echo "[1/5] Checking for cross-imports..."
if grep -r "from cloud\|import cloud" robot/ 2>/dev/null | grep -v ".pyc" | grep -v "__pycache__"; then
    echo "❌ ERROR: robot/ imports from cloud/"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ robot/ does not import from cloud/"
fi

if grep -rE "^[[:space:]]*(from robot|import robot)" cloud/ 2>/dev/null | grep -v ".pyc" | grep -v "__pycache__"; then
    echo "❌ ERROR: cloud/ imports from robot/"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ cloud/ does not import from robot/"
fi
echo ""

# Check 2: Robot start.sh runs robot code
echo "[2/5] Checking robot/start.sh..."
if grep -q "cd ../cloud" robot/start.sh; then
    echo "❌ ERROR: robot/start.sh changes to cloud directory"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ robot/start.sh stays in robot directory"
fi

if grep -q "uvicorn app.main:app" robot/start.sh; then
    echo "❌ ERROR: robot/start.sh runs cloud uvicorn server"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ robot/start.sh does not run cloud server"
fi

if grep -q "python main.py" robot/start.sh; then
    echo "✅ robot/start.sh runs robot/main.py"
else
    echo "❌ ERROR: robot/start.sh does not run robot/main.py"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 3: Separate virtual environments
echo "[3/5] Checking virtual environments..."
if grep -q "robot/venv" robot/start.sh; then
    echo "✅ robot/start.sh uses robot/venv"
elif grep -q "/venv\"" robot/start.sh | grep -v "robot/venv"; then
    echo "⚠️  WARNING: robot/start.sh uses shared venv (should use robot/venv)"
else
    echo "✅ robot/start.sh venv path looks correct"
fi

if [ -d "robot/venv" ]; then
    echo "✅ robot/venv exists"
else
    echo "⚠️  WARNING: robot/venv does not exist (run robot/setup.sh)"
fi

if [ -d "cloud/.venv" ]; then
    echo "✅ cloud/.venv exists"
else
    echo "⚠️  WARNING: cloud/.venv does not exist (run cloud/scripts/setup.sh)"
fi
echo ""

# Check 4: Robot config points to cloud
echo "[4/5] Checking robot config..."
if [ -f "robot/config.py" ]; then
    if grep -q "WS_PORT = 8765" robot/config.py; then
        echo "✅ robot connects to WebSocket port 8765"
    else
        echo "⚠️  WARNING: robot WebSocket port not found"
    fi
    
    if grep -q "SERVER_URL" robot/config.py; then
        echo "✅ robot has SERVER_URL configuration"
    else
        echo "❌ ERROR: robot missing SERVER_URL"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "❌ ERROR: robot/config.py not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 5: Cloud has both ports
echo "[5/5] Checking cloud config..."
if [ -f "cloud/config.py" ]; then
    if grep -q "API_PORT = 8000" cloud/config.py; then
        echo "✅ cloud REST API on port 8000"
    else
        echo "⚠️  WARNING: cloud API_PORT not found"
    fi
    
    if grep -q "WS_PORT = 8765" cloud/config.py; then
        echo "✅ cloud WebSocket on port 8765"
    else
        echo "⚠️  WARNING: cloud WS_PORT not found"
    fi
else
    echo "❌ ERROR: cloud/config.py not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Summary
echo "================================"
if [ $ERRORS -eq 0 ]; then
    echo "✅ ARCHITECTURE VERIFICATION PASSED"
    echo "================================"
    echo ""
    echo "The architecture is properly separated!"
    echo ""
    echo "Robot (Pi):"
    echo "  - Uses robot/ folder only"
    echo "  - Runs robot/main.py (WebSocket client)"
    echo "  - Connects TO cloud:8765"
    echo ""
    echo "Cloud (PC):"
    echo "  - Uses cloud/ folder only"
    echo "  - Runs cloud/main.py (servers)"
    echo "  - Provides port 8000 (REST) and 8765 (WebSocket)"
    exit 0
else
    echo "❌ ARCHITECTURE VERIFICATION FAILED"
    echo "================================"
    echo ""
    echo "Found $ERRORS error(s). Please review the issues above."
    echo ""
    echo "See ARCHITECTURE_FIXES.md for details on how to fix."
    exit 1
fi

