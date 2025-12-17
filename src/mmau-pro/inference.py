import argparse
import glob
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

warnings.simplefilter(
    "ignore"
)  # In any case, try to avoid warnings as much as possible.
warnings.simplefilter(
    "ignore", SyntaxWarning
)  # Use this instead if you can limit the type of warning.

seed = 42

set_seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def normalize_audio_path(value: Any) -> Optional[str]:
    """Normalize various audio_path container types to a single path string.

    Supports strings, lists/tuples (takes first), and numpy arrays coming from
    parquet deserialization. Returns None when no usable value is present.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) > 0 else None
    # Handle numpy arrays or array-like with .tolist()
    try:
        import numpy as _np  # type: ignore

        if isinstance(value, _np.ndarray):
            if value.size == 0:
                return None
            as_list = value.tolist()
            if isinstance(as_list, (list, tuple)):
                return as_list[0] if len(as_list) > 0 else None
            return str(as_list)
    except Exception:
        pass
    try:
        as_list = value.tolist()  # type: ignore[attr-defined]
        if isinstance(as_list, (list, tuple)):
            return as_list[0] if len(as_list) > 0 else None
        if isinstance(as_list, str):
            return as_list
    except Exception:
        pass
    return str(value)


def build_conversation(
    question_text: str, audio_path_or_url: Optional[str]
) -> List[Dict[str, Any]]:
    """Build a Qwen Omni conversation including audio if provided.

    Args:
        question_text: The user question/prompt text.
        audio_path_or_url: Optional audio path or URL.

    Returns:
        A list of chat messages suitable for Qwen Omni processor.
    """
    user_content: List[Dict[str, Any]] = []
    if audio_path_or_url:
        user_content.append({"type": "audio", "audio": audio_path_or_url})
    if question_text:
        user_content.append({"type": "text", "text": str(question_text)})

    conversation: List[Dict[str, Any]] = [
        {"role": "user", "content": user_content},
    ]
    return conversation


def get_mmau_pro_test_parquet_path() -> Optional[str]:
    """Return the most recent MMAU-Pro test.parquet path from HF hub cache.

    Looks under ~/.cache/huggingface/hub/datasets--gamma-lab-umd--MMAU-Pro/snapshots/*/test.parquet
    """
    root = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--gamma-lab-umd--MMAU-Pro/snapshots"
    )
    if not os.path.isdir(root):
        return None
    paths = glob.glob(os.path.join(root, "*", "test.parquet"))
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))


def load_input_dataframe(
    split: str, input_parquet: Optional[str]
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load test parquet directly from known HF hub cache (or --input).

    Args:
        split: Unused; kept for CLI compatibility.
        input_parquet: Optional direct parquet path to use.

    Returns:
        A tuple of (DataFrame, base_dir) where base_dir is used to resolve
        relative asset paths such as audio.

    Raises:
        RuntimeError: If no suitable input could be found.
    """
    if input_parquet and os.path.exists(input_parquet):
        abs_path = os.path.abspath(input_parquet)
        return pd.read_parquet(abs_path), os.path.dirname(abs_path)

    parquet_path = get_mmau_pro_test_parquet_path()
    if parquet_path and os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path), os.path.dirname(parquet_path)

    default_parquet = "/app/test.parquet"
    if os.path.exists(default_parquet):
        return pd.read_parquet(default_parquet), os.path.dirname(default_parquet)

    raise RuntimeError(
        "No input found. Provide --input parquet or ensure HF hub cache has MMAU-Pro test.parquet."
    )


def format_question(
    question_text: str,
    choices: Any,
    category: Optional[str],
    transcription: Optional[str] = None,
    task_classification: Optional[str] = None,
    task_identifier: Optional[str] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Construct the user prompt for a sample.

    This function builds a prompt string from the question and, when relevant,
    appends additional guidance based on the task type.

    - Multiple-choice: Appends options and asks to reply with the exact choice text.
    - Instruction following: Appends the provided ``transcription`` after the
      question so the model can follow explicit instructions contained in the
      audio intro.

    Args:
        question_text: The base question text.
        choices: Multiple-choice options if present; can be list/tuple/array.
        category: Dataset category (e.g., "sound", "open", "instruction following").
        transcription: Optional instruction string (used for instruction-following).
        task_classification: Optional classification label for instruction tasks.
        task_identifier: Optional identifier for instruction variants.
        extra_kwargs: Optional additional parameters for instruction tasks.

    Returns:
        The final prompt string to send to the model.
    """
    try:
        import numpy as _np  # type: ignore

        if isinstance(choices, _np.ndarray):
            choices = choices.tolist()
    except Exception:
        pass

    prompt_parts: List[str] = [str(question_text) if question_text is not None else ""]

    if isinstance(category, str) and category.strip().lower() == "open":
        pass
    elif (
        isinstance(category, str)
        and category.strip().lower() == "instruction following"
    ):
        prompt_parts = [str(transcription).strip()]
    else:  # 나머지
        choice_lines = "Options:\n"
        choice_lines += "\n".join(
            [f"{i + 1}) {str(choice)}" for i, choice in enumerate(choices)]
        )
        prompt_parts.append(f"\n\n{choice_lines}")

    return "".join(prompt_parts)


def generate_responses_with_audio(
    split: str,
    limit: Optional[int],
    output_path: str,
    input_parquet: Optional[str] = None,
    verbose: bool = False,
    verbose_n: int = 2,
    model: str = "qwen3-omni",
    attn_config: Optional[Dict[str, Any]] = None,
    use_white_noise: bool = False,
) -> None:
    """Generate model responses leveraging HF cache automatically.

    This function loads inputs from a provided parquet or auto-discovers a
    cached HF datasets Arrow split, and writes outputs in evaluator-compatible
    format (adds a 'model_output' column, preserves existing columns).
    """
    df, base_dir = load_input_dataframe(split=split, input_parquet=input_parquet)
    model = load_model(model, require_attention=bool(attn_config))

    num_samples = len(df)
    if limit is not None:
        num_samples = min(num_samples, max(0, int(limit)))

    df = df.iloc[:num_samples].copy()

    noise_path: Optional[str] = None
    if use_white_noise:
        root = detect_repo_root()
        candidate = resolve_repo_file("assets/noise/white-noise-358382.mp3", root)
        if not candidate.exists():
            raise FileNotFoundError(f"White noise file not found: {candidate}. Please ensure assets/noise/white-noise-358382.mp3 exists.")
        noise_path = str(candidate)
        print(f"Using white noise file for all samples: {noise_path}")

    outputs: List[str] = []
    debug_records: List[Dict[str, Any]] = []
    for idx in tqdm(range(num_samples), desc="Generating", ncols=100):
        sample = df.iloc[int(idx)].to_dict()

        question_text = str(sample.get("question", ""))
        choices = sample.get("choices")
        category = sample.get("category")
        transcription = sample.get("transcription")
        try:
            question_text = format_question(
                question_text,
                choices,
                category,
                transcription=transcription,
            )
        except Exception:
            pass

        audio_src = None
        ap_raw = sample.get("audio_path")
        ap_norm = normalize_audio_path(ap_raw)
        if isinstance(ap_norm, str) and ap_norm.strip():
            if re.match(r"^https?://", ap_norm) or os.path.isabs(ap_norm):
                audio_src = ap_norm
            else:
                audio_src = os.path.join(base_dir, ap_norm) if base_dir else ap_norm

        resolved_audio = noise_path if noise_path else audio_src
        conversation = build_conversation(question_text, resolved_audio)
        text_out = model.generate(conversation)
        outputs.append(text_out)

        if attn_config:
            try:
                attn = model.extract_attentions(
                    conversation,
                    layers=attn_config.get("layers"),
                )
                attn_config["root"].mkdir(parents=True, exist_ok=True)
                sample_dir = attn_config["root"] / f"sample_{idx:04d}"
                sample_id = sample.get("id") or f"mmau_{idx:04d}"
                meta = {
                    "model": attn_config["model"],
                    "sample_id": sample_id,
                    "benchmark": attn_config["benchmark"],
                    "split": attn_config["split"],
                    "prompt": question_text,
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
                        sample_id,
                    )
            except Exception as exc:
                print(
                    f"[WARN] Attention saving failed for sample {idx}: {exc}",
                    file=sys.stderr,
                )

        if verbose and idx < verbose_n:
            print("[DEBUG] id:", sample.get("id"))
            print("[DEBUG] category:", category)
            print("[DEBUG] resolved_audio:", resolved_audio)
            print(
                "[DEBUG] audio_exists:",
                bool(
                    isinstance(resolved_audio, str) and os.path.exists(resolved_audio)
                ),
            )
            print("[DEBUG] conversation:", conversation[0]["content"][1]["text"])
            print("[DEBUG] text_out:", text_out, "\n")
            try:
                debug_records.append(
                    {
                        "id": sample.get("id"),
                        "category": category,
                        "resolved_audio": resolved_audio,
                        "audio_exists": bool(
                            isinstance(resolved_audio, str)
                            and os.path.exists(resolved_audio)
                        ),
                        "conversation_text": conversation[0]["content"][1]["text"],
                        "text_out": text_out,
                    }
                )
            except Exception:
                pass

    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    df["model_output"] = outputs
    os.makedirs(output_path, exist_ok=True)
    df.to_parquet(os.path.join(output_path, "predictions.parquet"))
    print(f"Saved predictions to: {output_path}")

    # Save debug records to JSON if requested via verbose
    if verbose:
        try:
            dbg_path = os.path.join(output_path, "debug_info.json")
            os.makedirs(os.path.dirname(dbg_path), exist_ok=True)
            with open(dbg_path, "w", encoding="utf-8") as f:
                json.dump(debug_records, f, ensure_ascii=False, indent=2)
            print(f"Saved debug JSON to: {dbg_path}")
        except Exception as e:
            print(f"[WARN] Failed to write debug JSON: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen2.5-Omni predictions with audio and save to Parquet"
    )
    parser.add_argument(
        "--model", required=True, choices=["qwen2_5-omni", "qwen3-omni", "gemma3n"]
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Preferred split name for cache discovery (test/validation/train)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples for generation"
    )
    parser.add_argument(
        "--output",
        default="outputs/mmau-pro",
        help="Output directory to store results",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Optional input parquet. If omitted, attempts auto-discovery from HF datasets cache."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug info for the first few samples",
    )
    parser.add_argument(
        "--use-white-noise",
        action="store_true",
        help="Force all samples to use white noise audio instead of dataset audio.",
    )
    parser.add_argument(
        "--verbose_n",
        type=int,
        default=10000,
        help="How many samples to print debug info for",
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
        help="어텐션 저장 run 이름(미지정 시 timestamp).",
    )
    parser.add_argument(
        "--attn-plot-now",
        action="store_true",
        help="어텐션 저장 직후 이미지를 즉시 생성합니다.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, _ = parse_args()

    # Normalize output path relative to repo root
    root = detect_repo_root()
    args.output = str(normalize_output_path(args.output, root))

    attention_config = None
    if args.save_attn:
        run_name = args.attn_run_name or default_run_name()
        attention_config = {
            "root": Path(args.output) / "attn" / run_name,
            "run_name": run_name,
            "model": args.model,
            "benchmark": "mmau-pro",
            "split": args.split,
            "layers": args.attn_layers,
            "plot_now": args.attn_plot_now,
        }
    generate_responses_with_audio(
        split=args.split,
        limit=args.limit,
        output_path=args.output,
        input_parquet=args.input,
        verbose=args.verbose,
        verbose_n=args.verbose_n,
        model=args.model,
        attn_config=attention_config,
        use_white_noise=args.use_white_noise,
    )


if __name__ == "__main__":
    main()
