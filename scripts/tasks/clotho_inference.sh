#!/bin/bash
# Clotho Inference Task

MODEL=$1
OUTPUT_DIR=$2
RESULT_FILE=$3
ERROR_LOG="${OUTPUT_DIR}/clotho_infer.stderr.log"

echo "  [TASK] Running Clotho inference..."
: > "$ERROR_LOG"
START_TS=$(date +%s)

if uv run src/clotho/inference.py --split evaluation --model ${MODEL} --output_json_path ${OUTPUT_DIR}/clotho -t 2>"$ERROR_LOG"; then
    END_TS=$(date +%s); DURATION=$((END_TS-START_TS))
    echo "✅ Clotho inference completed (${DURATION}s)" >> "$RESULT_FILE"
    exit 0
else
    END_TS=$(date +%s); DURATION=$((END_TS-START_TS))
    echo "❌ Clotho inference failed (${DURATION}s)" >> "$RESULT_FILE"
    echo "--- Clotho inference error (last 50 lines) ---" >> "$RESULT_FILE"
    tail -n 50 "$ERROR_LOG" >> "$RESULT_FILE"
    exit 1
fi

