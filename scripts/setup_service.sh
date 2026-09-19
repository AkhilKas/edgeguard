#!/usr/bin/env bash
# Installs EdgeGuard as a systemd service on the UNO Q.
# Run once after cloning the repo and downloading the model.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_FILE="/etc/systemd/system/edgeguard.service"

sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=EdgeGuard — On-device prompt injection detection
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${PROJECT_DIR}/.env
ExecStart=$(which python3) -m edgeguard.api.server
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable edgeguard
sudo systemctl start edgeguard
echo "EdgeGuard service installed and started."
echo "Check status: sudo systemctl status edgeguard"
echo "View logs:    journalctl -u edgeguard -f"
