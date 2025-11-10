#!/bin/bash
# Inference Pipeline

MODEL=$1
OUTPUT_DIR=$2
RESULT_FILE=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[PIPELINE] Starting inference pipeline for $MODEL"

# 1. Environment setup (최하위 계층 호출)
bash "${SCRIPT_DIR}/../env/setup_inference.sh"

# 2. Run inference tasks (중간 계층 호출)
bash "${SCRIPT_DIR}/../tasks/mmau_pro_inference.sh" "$MODEL" "$OUTPUT_DIR" "$RESULT_FILE"
bash "${SCRIPT_DIR}/../tasks/clotho_inference.sh" "$MODEL" "$OUTPUT_DIR" "$RESULT_FILE"

echo "[PIPELINE] Inference pipeline completed"

