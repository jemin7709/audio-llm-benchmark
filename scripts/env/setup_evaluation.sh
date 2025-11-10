#!/bin/bash
# Environment Setup for Evaluation

echo "  [ENV] Setting up evaluation environment..."

uv remove transformers
uv add transformers==4.42.4

echo "  [ENV] Evaluation environment ready"

