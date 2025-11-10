#!/bin/bash
# Clotho Evaluation Only Pipeline

OUTPUT_DIR=$1
RESULT_FILE=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[PIPELINE] Starting Clotho evaluation"

# Evaluation environment setup + execution
bash "${SCRIPT_DIR}/../env/setup_evaluation.sh"
bash "${SCRIPT_DIR}/../tasks/clotho_evaluation.sh" "$OUTPUT_DIR" "$RESULT_FILE"

echo "[PIPELINE] Clotho evaluation completed"

