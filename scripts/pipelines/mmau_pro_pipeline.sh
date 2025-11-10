#!/bin/bash
# MMAU-Pro Only Pipeline (Inference + Evaluation)

MODEL=$1
OUTPUT_DIR=$2
RESULT_FILE=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[PIPELINE] Starting MMAU-Pro pipeline for $MODEL"

# 1. Inference environment setup + execution
bash "${SCRIPT_DIR}/../env/setup_inference.sh"
bash "${SCRIPT_DIR}/../tasks/mmau_pro_inference.sh" "$MODEL" "$OUTPUT_DIR" "$RESULT_FILE"

# 2. Evaluation environment setup + execution
bash "${SCRIPT_DIR}/../env/setup_evaluation.sh"
bash "${SCRIPT_DIR}/../tasks/mmau_pro_evaluation.sh" "$OUTPUT_DIR" "$RESULT_FILE"

echo "[PIPELINE] MMAU-Pro pipeline completed"

