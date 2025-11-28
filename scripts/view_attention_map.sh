#!/usr/bin/env bash

LOG_FILE="./outputs/run.log"
touch "$LOG_FILE"

run_and_log() {
  local cmd="$1"
  printf '%s\n' "$cmd" >> "$LOG_FILE"
  if eval "$cmd"; then
    printf 'STATUS: success (exit 0)\n' >> "$LOG_FILE"
  else
    local status=$?
    printf 'STATUS: failure (exit %d)\n' "$status" >> "$LOG_FILE"
    printf 'REASON: exit code %d\n' "$status" >> "$LOG_FILE"
  fi
}

run_and_log 'uv run lalm run mmau-pro --model gemma3n --save-attn --attn-run-name gemma_all --use-white-noise'
run_and_log 'uv run lalm run mmau-pro --model gemma3n --save-attn --attn-run-name gemma_all'
run_and_log 'uv run lalm run mmau-pro --model qwen2_5-omni --save-attn --attn-run-name qwen2_5-omni_all --use-white-noise'
run_and_log 'uv run lalm run mmau-pro --model qwen2_5-omni --save-attn --attn-run-name qwen2_5-omni_all'

run_and_log 'uv run -m src.utils.attention_plot --root /app/outputs/gemma3n_with_noise/mmau-pro/attn/gemma_all --workers 10'
run_and_log 'uv run -m src.utils.attention_plot --root /app/outputs/gemma3n/mmau-pro/attn/gemma_all --workers 10'
run_and_log 'uv run -m src.utils.attention_plot --root /app/outputs/qwen2_5-omni_with_noise/mmau-pro/attn/qwen2_5-omni_all --workers 10'
run_and_log 'uv run -m src.utils.attention_plot --root /app/outputs/qwen2_5-omni/mmau-pro/attn/qwen2_5-omni_all --workers 10'