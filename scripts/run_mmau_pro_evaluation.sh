#!/bin/bash
# MMAU-Pro Evaluation Only Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_mmau_pro_evaluation() {
    local MODEL=$1
    
    echo "=== Running MMAU-Pro evaluation for $MODEL ==="
    
    OUTPUT_DIR="./outputs/${MODEL}"
    RESULT_FILE="${OUTPUT_DIR}/result_mmau_pro_evaluation.txt"
    
    if [ ! -d "$OUTPUT_DIR" ] || [ ! -f "$OUTPUT_DIR/mmau-pro/predictions.parquet" ]; then
        echo "❌ Error: MMAU-Pro inference results not found for $MODEL"
        echo "   Please run inference first (run_mmau_pro_inference.sh $MODEL)"
        return 1
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    echo "MMAU-Pro Evaluation Results for $MODEL" > "$RESULT_FILE"
    echo "Started at: $(date)" >> "$RESULT_FILE"
    echo "=================================" >> "$RESULT_FILE"
    
    bash "${SCRIPT_DIR}/pipelines/mmau_pro_evaluation_only.sh" "$OUTPUT_DIR" "$RESULT_FILE"
    bash "${SCRIPT_DIR}/env/restore_env.sh"
    
    echo "Finished at: $(date)" >> "$RESULT_FILE"
    echo "=== Completed MMAU-Pro evaluation for $MODEL ==="
}

MODELS=("gemma3n" "qwen2_5-omni" "qwen3-omni")
if [ ! -z "$1" ]; then
    MODELS=("$1")
fi

for MODEL in "${MODELS[@]}"; do
    run_mmau_pro_evaluation "$MODEL"
done

