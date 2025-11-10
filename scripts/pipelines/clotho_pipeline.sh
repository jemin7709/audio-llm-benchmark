#!/bin/bash
# Clotho Only Pipeline (Inference + Evaluation)

MODEL=$1
OUTPUT_DIR=$2
RESULT_FILE=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[PIPELINE] Starting Clotho pipeline for $MODEL"

# 1. Inference environment setup + execution
bash "${SCRIPT_DIR}/../env/setup_inference.sh"
bash "${SCRIPT_DIR}/../tasks/clotho_inference.sh" "$MODEL" "$OUTPUT_DIR" "$RESULT_FILE"

# 2. Evaluation environment setup + execution
bash "${SCRIPT_DIR}/../env/setup_evaluation.sh"
bash "${SCRIPT_DIR}/../tasks/clotho_evaluation.sh" "$OUTPUT_DIR" "$RESULT_FILE"

echo "[PIPELINE] Clotho pipeline completed"

