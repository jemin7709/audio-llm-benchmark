#!/bin/bash
# Clotho Inference Only Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_clotho_inference() {
    local MODEL=$1
    
    echo "=== Running Clotho inference for $MODEL ==="
    
    OUTPUT_DIR="./outputs/${MODEL}"
    RESULT_FILE="${OUTPUT_DIR}/result_clotho_inference.txt"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Clotho Inference Results for $MODEL" > "$RESULT_FILE"
    echo "Started at: $(date)" >> "$RESULT_FILE"
    echo "=================================" >> "$RESULT_FILE"
    
    bash "${SCRIPT_DIR}/pipelines/clotho_inference_only.sh" "$MODEL" "$OUTPUT_DIR" "$RESULT_FILE"
    bash "${SCRIPT_DIR}/env/restore_env.sh"
    
    echo "Finished at: $(date)" >> "$RESULT_FILE"
    echo "=== Completed Clotho inference for $MODEL ==="
}

MODELS=("gemma3n" "qwen2_5-omni" "qwen3-omni")
if [ ! -z "$1" ]; then
    MODELS=("$1")
fi

for MODEL in "${MODELS[@]}"; do
    run_clotho_inference "$MODEL"
done

