#!/bin/bash
# Install ROVY Robot2 as a systemd service (runs on boot)

echo "Installing ROVY Robot2 service..."

# Copy service file
sudo cp rovy2.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable rovy2.service

echo ""
echo "✓ ROVY Robot2 service installed!"
echo ""
echo "Commands:"
echo "  sudo systemctl start rovy2    - Start now"
echo "  sudo systemctl stop rovy2     - Stop"
echo "  sudo systemctl restart rovy2  - Restart"
echo "  sudo systemctl status rovy2   - Check status"
echo "  journalctl -u rovy2 -f        - View logs"
echo ""
echo "The service will start automatically on boot."

