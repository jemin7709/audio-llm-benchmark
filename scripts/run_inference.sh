#!/bin/bash
# Inference Only Script
# Inference 작업만 독립적으로 실행

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_inference_only() {
    local MODEL=$1
    
    echo "=== Running inference for $MODEL ==="
    
    # 출력 디렉토리 준비
    OUTPUT_DIR="./outputs/${MODEL}"
    RESULT_FILE="${OUTPUT_DIR}/result_inference.txt"
    mkdir -p "$OUTPUT_DIR"
    
    # 결과 파일 초기화
    echo "Inference Results for $MODEL" > "$RESULT_FILE"
    echo "Started at: $(date)" >> "$RESULT_FILE"
    echo "=================================" >> "$RESULT_FILE"
    
    # Inference pipeline 실행
    bash "${SCRIPT_DIR}/pipelines/inference_pipeline.sh" "$MODEL" "$OUTPUT_DIR" "$RESULT_FILE"
    
    # 환경 복원
    bash "${SCRIPT_DIR}/env/restore_env.sh"
    
    echo "Finished at: $(date)" >> "$RESULT_FILE"
    echo "=== Completed inference for $MODEL ==="
}

# Entry Point
MODELS=("gemma3n" "qwen2_5-omni" "qwen3-omni")

# 특정 모델만 실행하려면: ./run_inference.sh gemma3n
if [ ! -z "$1" ]; then
    MODELS=("$1")
fi

for MODEL in "${MODELS[@]}"; do
    run_inference_only "$MODEL"
done

