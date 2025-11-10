#!/bin/bash
# Evaluation Pipeline

OUTPUT_DIR=$1
RESULT_FILE=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[PIPELINE] Starting evaluation pipeline"

# 1. Environment setup (최하위 계층 호출)
bash "${SCRIPT_DIR}/../env/setup_evaluation.sh"

# 2. Run evaluation tasks (중간 계층 호출)
bash "${SCRIPT_DIR}/../tasks/clotho_evaluation.sh" "$OUTPUT_DIR" "$RESULT_FILE"
bash "${SCRIPT_DIR}/../tasks/mmau_pro_evaluation.sh" "$OUTPUT_DIR" "$RESULT_FILE"

echo "[PIPELINE] Evaluation pipeline completed"

