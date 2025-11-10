#!/bin/bash
# Environment Setup for Inference

echo "  [ENV] Setting up inference environment..."

uv remove transformers 2>/dev/null
uv add transformers
uv add aac-metrics

uv run aac-metrics-download

uv remove transformers
uv add transformers

echo "  [ENV] Inference environment ready"

