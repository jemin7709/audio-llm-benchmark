#!/bin/bash

if [ -n "${HF_TOKEN:-}" ]; then
    echo "${HF_TOKEN}" | uvx huggingface-cli login --token >/dev/null 2>&1 || true
fi

if [ ! -d "${HF_HOME:-${HOME}/.cache/huggingface}/hub/datasets--woongvy--clotho-v2.1/" ]; then
    echo "Downloading Clotho dataset"
    uv run --project /app/envs/inference /app/src/utils/clotho_download.py -t || true
    cd ${HF_HOME:-${HOME}/.cache/huggingface}/hub/datasets--woongvy--clotho-v2.1/snapshots/*/

    unzip -n ./clotho.zip || true
    rm -f ./clotho.zip || true

    7za x ./clotho_audio_development.7z || true
    rm -f ./clotho_audio_development.7z || true

    7za x ./clotho_audio_validation.7z || true
    rm -f ./clotho_audio_validation.7z || true

    7za x ./clotho_audio_evaluation.7z || true
    rm -f ./clotho_audio_evaluation.7z || true
else
    echo "Clotho dataset already downloaded"
fi

cd /app

if [ ! -d "${HF_HOME:-${HOME}/.cache/huggingface}/hub/datasets--gamma-lab-umd--MMAU-Pro/" ]; then
    echo "Downloading MMAU-Pro dataset"
    uv run --project /app/envs/inference /app/src/utils/mmau-pro_download.py -t || true
    cd ${HF_HOME:-${HOME}/.cache/huggingface}/hub/datasets--gamma-lab-umd--MMAU-Pro/snapshots/*/

    unzip -n ./data.zip || true
    rm -f ./data.zip || true
else
    echo "MMAU-Pro dataset already downloaded"
fi
