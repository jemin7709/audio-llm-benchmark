#!/bin/bash
# Clotho Evaluation Only Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_clotho_evaluation() {
    local MODEL=$1
    
    echo "=== Running Clotho evaluation for $MODEL ==="
    
    OUTPUT_DIR="./outputs/${MODEL}"
    RESULT_FILE="${OUTPUT_DIR}/result_clotho_evaluation.txt"
    
    if [ ! -d "$OUTPUT_DIR" ] || [ ! -f "$OUTPUT_DIR/clotho/predictions.json" ]; then
        echo "❌ Error: Clotho inference results not found for $MODEL"
        echo "   Please run inference first (run_clotho_inference.sh $MODEL)"
        return 1
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    echo "Clotho Evaluation Results for $MODEL" > "$RESULT_FILE"
    echo "Started at: $(date)" >> "$RESULT_FILE"
    echo "=================================" >> "$RESULT_FILE"
    
    bash "${SCRIPT_DIR}/pipelines/clotho_evaluation_only.sh" "$OUTPUT_DIR" "$RESULT_FILE"
    bash "${SCRIPT_DIR}/env/restore_env.sh"
    
    echo "Finished at: $(date)" >> "$RESULT_FILE"
    echo "=== Completed Clotho evaluation for $MODEL ==="
}

MODELS=("gemma3n" "qwen2_5-omni" "qwen3-omni")
if [ ! -z "$1" ]; then
    MODELS=("$1")
fi

for MODEL in "${MODELS[@]}"; do
    run_clotho_evaluation "$MODEL"
done

