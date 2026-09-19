#!/usr/bin/env bash
# Downloads Qwen3.5-0.8B Q4_K_M (confirmed working on UNO Q QRB2210).
# Run this once on the device before starting EdgeGuard.
set -euo pipefail

MODEL_DIR="models"
MODEL_FILE="qwen3.5-0.8b-q4_k_m.gguf"
REMOTE_FILE="Qwen3.5-0.8B-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/${REMOTE_FILE}"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
  echo "Model already exists at ${MODEL_DIR}/${MODEL_FILE} — skipping download."
  exit 0
fi

echo "Downloading ${MODEL_FILE} (~537 MB)..."
curl -fL --progress-bar "${MODEL_URL}" -o "${MODEL_DIR}/${MODEL_FILE}"
echo "Done. Model saved to ${MODEL_DIR}/${MODEL_FILE}"
