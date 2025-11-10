#!/bin/bash
# MMAU-Pro Inference Task

MODEL=$1
OUTPUT_DIR=$2
RESULT_FILE=$3
ERROR_LOG="${OUTPUT_DIR}/mmau-pro_infer.stderr.log"

echo "  [TASK] Running MMAU-Pro inference..."
: > "$ERROR_LOG"
START_TS=$(date +%s)

if uv run src/mmau-pro/inference.py --split evaluation --verbose --model ${MODEL} --output ${OUTPUT_DIR}/mmau-pro -t 2>"$ERROR_LOG"; then
    END_TS=$(date +%s); DURATION=$((END_TS-START_TS))
    echo "✅ MMAU-Pro inference completed (${DURATION}s)" >> "$RESULT_FILE"
    exit 0
else
    END_TS=$(date +%s); DURATION=$((END_TS-START_TS))
    echo "❌ MMAU-Pro inference failed (${DURATION}s)" >> "$RESULT_FILE"
    echo "--- MMAU-Pro inference error (last 50 lines) ---" >> "$RESULT_FILE"
    tail -n 50 "$ERROR_LOG" >> "$RESULT_FILE"
    exit 1
fi

