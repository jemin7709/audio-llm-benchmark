#!/bin/bash
# Main Orchestration Script
# Inference + Evaluation 전체 워크플로우 실행
# 
# Usage:
#   ./run.sh [model_name]           # 전체 워크플로우 (inference + evaluation)
#   ./run_inference.sh [model_name] # inference만 실행
#   ./run_evaluation.sh [model_name] # evaluation만 실행 (inference 결과 필요)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_model_evaluation() {
    local MODEL=$1
    
    echo "=== Running evaluation for $MODEL ==="
    
    # 출력 디렉토리 준비
    OUTPUT_DIR="./outputs/${MODEL}"
    RESULT_FILE="${OUTPUT_DIR}/result.txt"
    mkdir -p "$OUTPUT_DIR"
    
    # 결과 파일 초기화
    echo "Evaluation Results for $MODEL" > "$RESULT_FILE"
    echo "Started at: $(date)" >> "$RESULT_FILE"
    echo "=================================" >> "$RESULT_FILE"
    
    # Inference pipeline 실행 (상위 계층 스크립트 호출)
    bash "${SCRIPT_DIR}/pipelines/inference_pipeline.sh" "$MODEL" "$OUTPUT_DIR" "$RESULT_FILE"
    
    # Evaluation pipeline 실행 (상위 계층 스크립트 호출)
    bash "${SCRIPT_DIR}/pipelines/evaluation_pipeline.sh" "$OUTPUT_DIR" "$RESULT_FILE"
    
    # 환경 복원 (최하위 계층 스크립트 호출)
    bash "${SCRIPT_DIR}/env/restore_env.sh"
    
    echo "Finished at: $(date)" >> "$RESULT_FILE"
    echo "=== Completed $MODEL ==="
}

# Entry Point
MODELS=("gemma3n" "qwen2_5-omni" "qwen3-omni")

# 특정 모델만 실행하려면: ./run.sh gemma3n
if [ ! -z "$1" ]; then
    MODELS=("$1")
fi

for MODEL in "${MODELS[@]}"; do
    run_model_evaluation "$MODEL"
done
