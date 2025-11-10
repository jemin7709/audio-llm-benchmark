#!/bin/bash
# Clotho Inference Only Pipeline

MODEL=$1
OUTPUT_DIR=$2
RESULT_FILE=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[PIPELINE] Starting Clotho inference for $MODEL"

# Inference environment setup + execution
bash "${SCRIPT_DIR}/../env/setup_inference.sh"
bash "${SCRIPT_DIR}/../tasks/clotho_inference.sh" "$MODEL" "$OUTPUT_DIR" "$RESULT_FILE"

echo "[PIPELINE] Clotho inference completed"

