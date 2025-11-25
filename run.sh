CUDA_VISIBLE_DEVICES=5 uv run clair_a.py --model google/gemma-3n-E4B-it
status1=$?

CUDA_VISIBLE_DEVICES=5 uv run clair_a.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct
status2=$?

CUDA_VISIBLE_DEVICES=5 uv run clair_a.py --model Qwen/Qwen2.5-Omni-7B
status3=$?

echo "실행 결과:"
[ $status1 -eq 0 ] && echo "✓ google/gemma-3n-E4B-it: 성공" || echo "✗ google/gemma-3n-E4B-it: 실패"
[ $status2 -eq 0 ] && echo "✓ Qwen/Qwen3-Omni-30B-A3B-Instruct: 성공" || echo "✗ Qwen/Qwen3-Omni-30B-A3B-Instruct: 실패"
[ $status3 -eq 0 ] && echo "✓ Qwen/Qwen2.5-Omni-7B: 성공" || echo "✗ Qwen/Qwen2.5-Omni-7B: 실패"