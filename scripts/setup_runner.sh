#!/usr/bin/env bash
# Registers a GitHub Actions self-hosted runner on the UNO Q.
# Run once on the device. After this, pushes to main trigger on-device deployment.
#
# Usage:
#   export GITHUB_REPO="your-username/edgeguard"
#   export GITHUB_TOKEN="your-runner-registration-token"   # from Settings > Actions > Runners
#   bash scripts/setup_runner.sh
set -euo pipefail

RUNNER_VERSION="2.316.1"
RUNNER_DIR="${HOME}/actions-runner"

if [ -z "${GITHUB_REPO:-}" ] || [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "ERROR: Set GITHUB_REPO and GITHUB_TOKEN before running."
  exit 1
fi

mkdir -p "${RUNNER_DIR}" && cd "${RUNNER_DIR}"

# Download runner
curl -sL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz" \
  | tar xz

# Configure
./config.sh \
  --url "https://github.com/${GITHUB_REPO}" \
  --token "${GITHUB_TOKEN}" \
  --name "uno-q-$(hostname)" \
  --labels "self-hosted,uno-q,arm64" \
  --unattended

# Install as a systemd service so it survives reboots
sudo ./svc.sh install
sudo ./svc.sh start

echo "Runner registered and started. Check GitHub > Settings > Actions > Runners."
