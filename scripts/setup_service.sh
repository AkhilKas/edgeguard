#!/usr/bin/env bash
# Installs EdgeGuard as a systemd service on the UNO Q.
# Run once after cloning the repo, creating .venv (pip install -e ".[dev]"),
# and downloading the model.
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
# No EnvironmentFile= here on purpose: systemd's parser doesn't strip
# inline "#" comments the way our own dotenv loading does, so lines like
# "MCU_ENABLED=true   # set false in CI" would load the comment text as
# part of the value and fail pydantic validation. Settings() already reads
# .env itself (relative to WorkingDirectory), so this isn't needed anyway.
ExecStart=${PROJECT_DIR}/.venv/bin/python3 -m edgeguard.api.server
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
