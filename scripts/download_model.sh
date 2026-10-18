#!/usr/bin/env bash
# Downloads LFM2.5-350M Q4_K_M (confirmed working on UNO Q QRB2210; chosen
# over Qwen3.5-0.8B after a head-to-head comparison - smaller, consistently
# faster, and didn't show Qwen's occasional runaway "thinking" generations).
# Run this once on the device before starting EdgeGuard.
set -euo pipefail

MODEL_DIR="models"
MODEL_FILE="lfm2.5-350m-q4_k_m.gguf"
REMOTE_FILE="LFM2.5-350M-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/${REMOTE_FILE}"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
  echo "Model already exists at ${MODEL_DIR}/${MODEL_FILE} — skipping download."
  exit 0
fi

echo "Downloading ${MODEL_FILE} (~219 MB)..."
curl -fL --progress-bar "${MODEL_URL}" -o "${MODEL_DIR}/${MODEL_FILE}"
echo "Done. Model saved to ${MODEL_DIR}/${MODEL_FILE}"
