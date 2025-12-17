#!/bin/bash
# Guard script: Verify vLLM is installed in envs/inference without forcing additional installation
set -euo pipefail

echo "Checking vLLM availability in envs/inference..."

if uv run --project envs/inference python -c "import vllm; print(f'vLLM {vllm.__version__} available')" 2>/dev/null; then
    echo "✓ vLLM successfully verified"
    exit 0
else
    echo "✗ vLLM verification failed"
    echo "Please ensure envs/inference is properly synced:"
    echo "  uv sync --project envs/inference --frozen"
    exit 1
fi

