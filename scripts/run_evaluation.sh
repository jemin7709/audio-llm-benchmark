#!/bin/bash
# Evaluation Only Script
# Evaluation 작업만 독립적으로 실행 (inference 결과가 이미 있어야 함)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_evaluation_only() {
    local MODEL=$1
    
    echo "=== Running evaluation for $MODEL ==="
    
    # 출력 디렉토리 확인
    OUTPUT_DIR="./outputs/${MODEL}"
    RESULT_FILE="${OUTPUT_DIR}/result_evaluation.txt"
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "❌ Error: Output directory $OUTPUT_DIR does not exist."
        echo "   Please run inference first (run_inference.sh $MODEL)"
        return 1
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    # 결과 파일 초기화
    echo "Evaluation Results for $MODEL" > "$RESULT_FILE"
    echo "Started at: $(date)" >> "$RESULT_FILE"
    echo "=================================" >> "$RESULT_FILE"
    
    # Evaluation pipeline 실행
    bash "${SCRIPT_DIR}/pipelines/evaluation_pipeline.sh" "$OUTPUT_DIR" "$RESULT_FILE"
    
    # 환경 복원
    bash "${SCRIPT_DIR}/env/restore_env.sh"
    
    echo "Finished at: $(date)" >> "$RESULT_FILE"
    echo "=== Completed evaluation for $MODEL ==="
}

# Entry Point
MODELS=("gemma3n" "qwen2_5-omni" "qwen3-omni")

# 특정 모델만 실행하려면: ./run_evaluation.sh gemma3n
if [ ! -z "$1" ]; then
    MODELS=("$1")
fi

for MODEL in "${MODELS[@]}"; do
    run_evaluation_only "$MODEL"
done

