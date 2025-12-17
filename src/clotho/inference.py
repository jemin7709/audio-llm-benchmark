#!/usr/bin/env python3
# clotho_eval.py
"""
(Part 1) Generate Predictions for Clotho-v2 using Audio Flamingo 3.

This script runs in the 'af3' conda environment.
It loads the Clotho-v2 dataset, generates captions for each audio file using the
model's CLI, and saves the predictions along with ground-truth references
to a single JSON file.
"""

import argparse
import glob
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    set_seed,
)
from src.models import load_model
from src.utils.attention_io import default_run_name, save_attention_bundle
from src.utils.attention_plot import save_sample_plots
from src.utils.paths import detect_repo_root, normalize_output_path, resolve_repo_file

seed = 42

set_seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def extract_clotho_dataset(zip_path: str, extract_to: str) -> str:
    """Extracts the Clotho-v2 dataset from a .zip file."""
    # if not os.path.exists(extract_to):
    #     print(f"Extracting {zip_path} to {extract_to}...")
    #     os.makedirs(extract_to, exist_ok=True)
    #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #         zip_ref.extractall(extract_to)
    #     print("Extraction complete.")
    # else:
    #     print(f"Directory {extract_to} already exists. Skipping extraction.")

    # Handle common case where zip extracts to a single sub-folder
    extracted_items = os.listdir(extract_to)
    if len(extracted_items) == 1 and os.path.isdir(
        os.path.join(extract_to, extracted_items[0])
    ):
        return os.path.join(extract_to, extracted_items[0])
    return extract_to


def load_clotho_data(clotho_base_path: str, split: str = "evaluation") -> Dict:
    """Loads the Clotho-v2 dataset for a specific split."""
    if split == "all":
        csv_file1 = os.path.join(clotho_base_path, "clotho_captions_development.csv")
        audio_dir1 = os.path.join(clotho_base_path, "development")
        csv_file2 = os.path.join(clotho_base_path, "clotho_captions_validation.csv")
        audio_dir2 = os.path.join(clotho_base_path, "validation")
        csv_file3 = os.path.join(clotho_base_path, "clotho_captions_evaluation.csv")
        audio_dir3 = os.path.join(clotho_base_path, "evaluation")

        csv_files = [csv_file1, csv_file2, csv_file3]
        audio_dirs = [audio_dir1, audio_dir2, audio_dir3]

        print("Loading CSV from: dev, val, eval")
        print(f"Loading Audio from: {audio_dirs}")
    else:
        csv_file = os.path.join(clotho_base_path, f"clotho_captions_{split}.csv")
        audio_dir = os.path.join(clotho_base_path, split)

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        audio_dirs = [audio_dir]
        print(f"Loading CSV from: {csv_file}")
        print(f"Loading Audio from: {audio_dirs}")

    if split == "all":
        df = {}
        lens = []
        for csv_file, audio_dir in zip(csv_files, audio_dirs):
            df[csv_file] = pd.read_csv(csv_file)
            lens.append(len(df[csv_file]))
        df = pd.concat(df)
        print(f"Total samples: {len(df)}")
        print(
            f"Dev samples: {lens[0]}, Val samples: {lens[1]}, Eval samples: {lens[2]}"
        )
    else:
        df = pd.read_csv(csv_file)
    data = {}
    missing_audio_count = 0

    for _, row in df.iterrows():
        audio_file = row["file_name"]
        audio_path = None
        for base_dir in audio_dirs:
            candidate = os.path.join(base_dir, audio_file)
            if os.path.exists(candidate):
                audio_path = candidate
                break
        if audio_path is None:
            missing_audio_count += 1
            print(f"Warning: {audio_file} not found on disk.")
            continue

        captions = [
            str(row[f"caption_{i}"]).strip()
            for i in range(1, 6)
            if f"caption_{i}" in row and pd.notna(row[f"caption_{i}"])
        ]

        if captions:
            data[audio_file] = {"audio_path": audio_path, "references": captions}

    # if missing_audio_count > 0:
    print(f"Warning: {missing_audio_count} audio files from CSV not found on disk.")

    print(f"Successfully loaded {len(data)} samples for the '{split}' split.")
    return data


def generate_predictions(
    data: Dict,
    model,
    prompt: str,
    timeout: int,
    infer_script_path: str,
    attn_config: Optional[Dict[str, Any]] = None,
    use_white_noise: bool = False,
) -> Dict[str, str]:
    """Generates predictions for each audio file using the Audio Flamingo 3 CLI."""
    predictions = {}
    failed_samples = []

    noise_path: Optional[str] = None
    if use_white_noise:
        root = detect_repo_root()
        candidate = resolve_repo_file("assets/noise/white-noise-358382.mp3", root)
        if not candidate.exists():
            raise FileNotFoundError(f"White noise file not found: {candidate}. Please ensure assets/noise/white-noise-358382.mp3 exists.")
        noise_path = str(candidate)
        print(
            f"Generating predictions for {len(data)} samples using white noise file: {noise_path}"
        )
    else:
        print(f"Generating predictions for {len(data)} samples using dataset audio.")

    # if not os.path.exists(infer_script_path):
    #     raise FileNotFoundError(f"Inference script not found at {infer_script_path}. Please provide correct path.")

    for idx, (audio_file, info) in enumerate(
        tqdm(data.items(), desc="Generating Captions")
    ):
        try:
            resolved_audio = noise_path or info["audio_path"]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": resolved_audio,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_out = model.generate(conversation)
            predictions[audio_file] = text_out

            print("[DEBUG] conversation:", conversation)
            print("[DEBUG] audio path:", resolved_audio)
            print("[DEBUG] answer:", info["references"][0])
            print("[DEBUG] text_out:", text_out, "\n")

            if attn_config:
                try:
                    attn = model.extract_attentions(
                        conversation,
                        layers=attn_config.get("layers"),
                    )
                    attn_config["root"].mkdir(parents=True, exist_ok=True)
                    sample_dir = attn_config["root"] / f"sample_{idx:04d}"
                    meta = {
                        "model": attn_config["model"],
                        "sample_id": audio_file,
                        "benchmark": attn_config["benchmark"],
                        "split": attn_config["split"],
                        "prompt": attn_config["prompt"],
                        "attn_run_name": attn_config["run_name"],
                        "layers": attn["layers"],
                    }
                    save_attention_bundle(
                        attn["attentions"],
                        attn["tokens"],
                        meta,
                        sample_dir,
                    )
                    if attn_config.get("plot_now"):
                        save_sample_plots(
                            attn["attentions"],
                            attn["tokens"],
                            attn["layers"],
                            sample_dir,
                            audio_file,
                        )
                except Exception as exc:
                    print(
                        f"[WARN] Attention saving failed for {audio_file}: {exc}",
                        file=sys.stderr,
                    )
        except Exception as e:
            print(f"\nAn unexpected error occurred with {audio_file}: {e}\n")
            print(traceback.format_exc())
            failed_samples.append(audio_file)
            predictions[audio_file] = "ERROR"

    if failed_samples:
        print(f"Failed to generate captions for {len(failed_samples)} samples.")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Part 1: Generate predictions for Clotho-v2."
    )
    parser.add_argument(
        "--clotho_zip_path",
        type=str,
        default="~/.cache/huggingface/hub/datasets--woongvy--clotho-v2.1/snapshots/*/clotho.zip",
        help="Path to the Clotho-v2 dataset zip file.",
    )
    parser.add_argument(
        "--model", required=True, choices=["qwen2_5-omni", "qwen3-omni", "gemma3n"]
    )
    parser.add_argument(
        "--split", type=str, default="all", help="Dataset split to evaluate."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the sound in detail.",
        help="Prompt to use for captioning.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Timeout for each model inference call in seconds.",
    )
    parser.add_argument(
        "--infer_script_path",
        type=str,
        default="audio-flamingo/llava/cli/infer_audio.py",
        help="Path to the inference script.",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        default="outputs/clotho",
        help="Directory to save predictions.json and related files.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to process for a quick test.",
    )
    parser.add_argument(
        "--use-white-noise",
        action="store_true",
        help="Force using the bundled white noise audio instead of original audio.",
    )
    parser.add_argument(
        "--save-attn",
        action="store_true",
        help="Attention map을 저장합니다.",
    )
    parser.add_argument(
        "--attn-layers",
        nargs="+",
        type=int,
        help="저장할 레이어 인덱스(0 기반).",
    )
    parser.add_argument(
        "--attn-run-name",
        type=str,
        help="어텐션 저장에 사용할 run 이름(미지정 시 timestamp).",
    )
    parser.add_argument(
        "--attn-plot-now",
        action="store_true",
        help="어텐션 저장 직후 이미지까지 함께 생성합니다.",
    )
    # Device selection removed; rely on default behavior

    args, _ = parser.parse_known_args()

    # Normalize output path relative to repo root
    root = detect_repo_root()
    args.output_json_path = str(normalize_output_path(args.output_json_path, root))

    # Expand a single directory-level '*' in the path (e.g., snapshots/*/clotho.zip)
    def _expand_single_star_dir(path_pattern: str) -> str:
        expanded = os.path.expanduser(path_pattern)
        parts = expanded.split(os.sep)
        for idx, comp in enumerate(parts):
            if "*" in comp or "?" in comp or "[" in comp:
                prefix = os.sep.join(parts[:idx])
                suffix = os.sep.join(parts[idx + 1 :])
                pattern = os.path.join(prefix, comp)
                candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
                if not candidates:
                    raise FileNotFoundError(f"No directory matches pattern: {pattern}")
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return os.path.join(candidates[0], suffix)
        return expanded

    # 1. Load Dataset
    try:
        resolved_zip_path = _expand_single_star_dir(args.clotho_zip_path)
        extract_dir = Path(resolved_zip_path).parent
        clotho_base_path = extract_clotho_dataset(resolved_zip_path, str(extract_dir))
        clotho_data = load_clotho_data(clotho_base_path, args.split)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Sample Data if requested
    if args.sample_size and args.sample_size < len(clotho_data):
        print(f"Using a random sample of {args.sample_size} audio files.")
        sample_keys = random.sample(list(clotho_data.keys()), args.sample_size)
        clotho_data = {k: clotho_data[k] for k in sample_keys}

    attention_config = None
    if args.save_attn:
        run_name = args.attn_run_name or default_run_name()
        attention_config = {
            "root": Path(args.output_json_path) / "attn" / run_name,
            "run_name": run_name,
            "model": args.model,
            "benchmark": "clotho",
            "split": args.split,
            "prompt": args.prompt,
            "layers": args.attn_layers,
            "plot_now": args.attn_plot_now,
        }

    # 3. Generate Predictions
    model = load_model(args.model, require_attention=args.save_attn)
    predictions = generate_predictions(
        clotho_data,
        model,
        args.prompt,
        args.timeout,
        args.infer_script_path,
        attn_config=attention_config,
        use_white_noise=args.use_white_noise,
    )

    # 4. Combine predictions with references and save
    output_data = []
    for audio_file, data_info in clotho_data.items():
        output_data.append(
            {
                "audio_id": audio_file,
                "prediction": predictions.get(audio_file, ""),  # Use .get for safety
                "references": data_info["references"],
            }
        )

    os.makedirs(args.output_json_path, exist_ok=True)
    with open(os.path.join(args.output_json_path, "predictions.json"), "w") as f:
        json.dump(output_data, f, indent=4)

    print(
        f"\nPredictions and references successfully saved to {os.path.join(args.output_json_path, 'predictions.json')}"
    )


if __name__ == "__main__":
    main()
