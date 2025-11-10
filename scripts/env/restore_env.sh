#!/bin/bash
# Restore Default Environment

echo "  [ENV] Restoring default environment..."

uv remove transformers
uv add transformers

echo "  [ENV] Default environment restored"

