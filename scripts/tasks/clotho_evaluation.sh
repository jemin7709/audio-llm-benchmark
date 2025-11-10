#!/bin/bash
# Clotho Evaluation Task

OUTPUT_DIR=$1
RESULT_FILE=$2
ERROR_LOG="${OUTPUT_DIR}/clotho_eval.stderr.log"

echo "  [TASK] Running Clotho evaluation..."
: > "$ERROR_LOG"
START_TS=$(date +%s)

if uv run src/clotho/evaluation.py --input_json_path ${OUTPUT_DIR}/clotho/predictions.json --output_json_path ${OUTPUT_DIR}/clotho/scores.json -t 2>"$ERROR_LOG"; then
    END_TS=$(date +%s); DURATION=$((END_TS-START_TS))
    echo "✅ Clotho evaluation completed (${DURATION}s)" >> "$RESULT_FILE"
    exit 0
else
    END_TS=$(date +%s); DURATION=$((END_TS-START_TS))
    echo "❌ Clotho evaluation failed (${DURATION}s)" >> "$RESULT_FILE"
    echo "--- Clotho evaluation error (last 50 lines) ---" >> "$RESULT_FILE"
    tail -n 50 "$ERROR_LOG" >> "$RESULT_FILE"
    exit 1
fi

