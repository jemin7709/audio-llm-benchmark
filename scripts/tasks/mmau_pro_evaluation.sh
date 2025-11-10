#!/bin/bash
# MMAU-Pro Evaluation Task

OUTPUT_DIR=$1
RESULT_FILE=$2
ERROR_LOG="${OUTPUT_DIR}/mmau-pro_eval.stderr.log"

echo "  [TASK] Running MMAU-Pro evaluation..."
: > "$ERROR_LOG"
START_TS=$(date +%s)

if uv run src/mmau-pro/evaluation.py ${OUTPUT_DIR}/mmau-pro/predictions.parquet -t 2>"$ERROR_LOG"; then
    END_TS=$(date +%s); DURATION=$((END_TS-START_TS))
    echo "✅ MMAU-Pro evaluation completed (${DURATION}s)" >> "$RESULT_FILE"
    exit 0
else
    END_TS=$(date +%s); DURATION=$((END_TS-START_TS))
    echo "❌ MMAU-Pro evaluation failed (${DURATION}s)" >> "$RESULT_FILE"
    echo "--- MMAU-Pro evaluation error (last 50 lines) ---" >> "$RESULT_FILE"
    tail -n 50 "$ERROR_LOG" >> "$RESULT_FILE"
    exit 1
fi

