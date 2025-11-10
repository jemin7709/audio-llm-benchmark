#!/bin/bash
# MMAU-Pro Only Script
# MMAU-Pro 벤치마크만 실행 (inference + evaluation)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_mmau_pro_benchmark() {
    local MODEL=$1
    
    echo "=== Running MMAU-Pro benchmark for $MODEL ==="
    
    # 출력 디렉토리 준비
    OUTPUT_DIR="./outputs/${MODEL}"
    RESULT_FILE="${OUTPUT_DIR}/result_mmau_pro.txt"
    mkdir -p "$OUTPUT_DIR"
    
    # 결과 파일 초기화
    echo "MMAU-Pro Benchmark Results for $MODEL" > "$RESULT_FILE"
    echo "Started at: $(date)" >> "$RESULT_FILE"
    echo "=================================" >> "$RESULT_FILE"
    
    # MMAU-Pro pipeline 실행
    bash "${SCRIPT_DIR}/pipelines/mmau_pro_pipeline.sh" "$MODEL" "$OUTPUT_DIR" "$RESULT_FILE"
    
    # 환경 복원
    bash "${SCRIPT_DIR}/env/restore_env.sh"
    
    echo "Finished at: $(date)" >> "$RESULT_FILE"
    echo "=== Completed MMAU-Pro benchmark for $MODEL ==="
}

# Entry Point
MODELS=("gemma3n" "qwen2_5-omni" "qwen3-omni")

# 특정 모델만 실행하려면: ./run_mmau_pro.sh gemma3n
if [ ! -z "$1" ]; then
    MODELS=("$1")
fi

for MODEL in "${MODELS[@]}"; do
    run_mmau_pro_benchmark "$MODEL"
done

