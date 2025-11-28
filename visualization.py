"""
Gemma3N attention extraction & visualization entry-point.

Example:
    uv run python visualization.py --prompt "Hello" --layers 0 1 --limit-samples 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.models.gemma3n import Gemma3N
from src.utils.attention_utils import layer_list_to_array, mean_over_layers, stack_batch
from src.utils.attention_io import default_run_name, save_attention_bundle
from src.utils.attention_plot import plot_attention_map, plot_layer_grid


def load_attention(
    sample_dir: Path,
) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
    """
    저장된 어텐션 번들을 로드합니다.

    Args:
        sample_dir: `attn.npy`, `tokens.json`, `meta.json`이 있는 디렉토리 경로.

    Returns:
        (layers, tokens, meta) 순서로 반환합니다.
        - layers: 레이어별 어텐션 행렬 리스트(shape: S x S)
        - tokens: 토큰 문자열 리스트
        - meta: 메타데이터 딕셔너리
    """
    sample_dir = Path(sample_dir)
    attn_path = sample_dir / "attn.npy"
    tokens_path = sample_dir / "tokens.json"
    meta_path = sample_dir / "meta.json"

    layers = np.load(attn_path, allow_pickle=False)
    with tokens_path.open("r", encoding="utf-8") as file:
        tokens = json.load(file)
    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)
    layer_list = [np.asarray(layer, dtype=np.float32) for layer in layers]
    return layer_list, tokens, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and visualize Gemma3N attention maps."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="단일 사용자 메시지. --input-jsonl과 동시에 사용 가능.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help="JSONL 경로. 각 줄은 {'messages': [...]} 또는 {'prompt': '...'} 형식.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3n-E4B-it",
        help="Hugging Face model id (기본: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="모델을 올릴 디바이스(e.g., cuda:0, cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="모델 dtype (기본: auto).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["all"],
        help="시각화할 레이어 인덱스(예: --layers 0 5 8). 'all'이면 전체.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        help="상위 N개의 레이어만 처리(레이어 인덱스를 따로 주지 않은 경우).",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="처리할 샘플 수 상한.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/attn"),
        help="결과 파일 루트 디렉토리.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="출력 하위 폴더 이름. 미지정 시 timestamp 사용.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="모델 초기화 시 사용할 시드(현재는 저장 목적으로만 사용).",
    )
    return parser


def resolve_layers(
    layer_args: Sequence[str], max_layers: Optional[int]
) -> Optional[List[int]]:
    if not layer_args:
        return None
    normalized = [token.lower() for token in layer_args]
    if len(normalized) == 1 and normalized[0] == "all":
        if max_layers is None:
            return None
        return list(range(max_layers))

    indices: List[int] = []
    for token in normalized:
        if token == "all":
            continue
        try:
            indices.append(int(token))
        except ValueError as err:
            raise ValueError(f"레이어 인덱스는 정수여야 합니다: '{token}'") from err

    if max_layers is not None:
        indices = indices[:max_layers]
    return sorted(set(indices))


def load_samples(
    prompt: Optional[str], jsonl_path: Optional[Path]
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    sample_idx = 0

    if prompt:
        samples.append(
            {
                "id": f"prompt_{sample_idx:04d}",
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        sample_idx += 1

    if jsonl_path:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"입력 JSONL을 찾을 수 없습니다: {jsonl_path}")
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                sample_id = None
                messages = None
                if isinstance(record, str):
                    messages = [{"role": "user", "content": record}]
                elif isinstance(record, dict):
                    sample_id = record.get("id")
                    if "messages" in record:
                        messages = record["messages"]
                    elif "prompt" in record:
                        messages = [{"role": "user", "content": record["prompt"]}]
                if not messages:
                    raise ValueError(
                        "각 JSONL 라인은 문자열 또는 {'messages': [...]} / {'prompt': '...'} 여야 합니다."
                    )
                if not sample_id:
                    sample_id = f"jsonl_{sample_idx:04d}"
                samples.append({"id": sample_id, "messages": messages})
                sample_idx += 1

    return samples


def next_run_name(run_name: Optional[str]) -> str:
    return run_name or default_run_name()


def align_samples(samples: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    seq_lengths = [sample.shape[-1] for sample in samples]
    min_seq = min(seq_lengths)
    if len(set(seq_lengths)) == 1:
        return samples, min_seq
    aligned = [sample[:, :min_seq, :min_seq] for sample in samples]
    return aligned, min_seq


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.prompt and not args.input_jsonl:
        parser.error(
            "적어도 하나의 입력 옵션(--prompt 또는 --input-jsonl)이 필요합니다."
        )
    layer_indices = resolve_layers(args.layers, args.max_layers)
    samples = load_samples(args.prompt, args.input_jsonl)
    if args.limit_samples is not None:
        samples = samples[: args.limit_samples]
    if not samples:
        parser.error("처리할 샘플이 없습니다.")

    run_name = next_run_name(args.run_name)
    run_root = args.output_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading Gemma3N model '{args.model}' on {args.device}...")
    model = Gemma3N(
        name=args.model, dtype=args.dtype, device=args.device, seed=args.seed
    )

    aggregated_arrays: List[np.ndarray] = []
    sample_stats: Dict[str, Any] = {}
    reference_tokens: Optional[List[str]] = None
    aligned_length: Optional[int] = None

    for sample in samples:
        attn_pack = model.extract_attentions(
            messages=sample["messages"],
            layers=layer_indices,
        )
        layer_order = attn_pack["layers"]
        layer_array = layer_list_to_array(attn_pack["attentions"])
        tokens = attn_pack["tokens"]
        sample_dir = run_root / sample["id"]
        meta = {
            "model": args.model,
            "run_name": run_name,
            "sample_id": sample["id"],
            "layers": layer_order,
            "messages": sample["messages"],
        }
        save_attention_bundle(
            attn_pack["attentions"],
            tokens,
            meta,
            sample_dir,
        )

        for idx, matrix in zip(layer_order, layer_array):
            plot_attention_map(
                matrix,
                tokens,
                title=f"{sample['id']} · Layer {idx}",
                output_path=sample_dir / f"layer_{idx:02d}.jpg",
            )

        mean_matrix = mean_over_layers(layer_array)
        plot_attention_map(
            mean_matrix,
            tokens,
            title=f"{sample['id']} · Layer mean",
            output_path=sample_dir / "layer_mean.jpg",
        )

        aggregated_arrays.append(layer_array)
        sample_stats[sample["id"]] = {
            "num_layers": layer_array.shape[0],
            "seq_len": layer_array.shape[-1],
            "tokens": len(tokens),
        }
        if reference_tokens is None:
            reference_tokens = list(tokens)

    if aggregated_arrays:
        seq_lengths = [arr.shape[-1] for arr in aggregated_arrays]
        aligned_arrays, min_seq = align_samples(aggregated_arrays)
        aligned_length = min_seq
        stacked = stack_batch(aligned_arrays)
        global_layer_mean = stacked.mean(axis=0)
        global_mean = global_layer_mean.mean(axis=0)

        if len(set(seq_lengths)) > 1:
            print(
                "[WARN] 샘플별 토큰 길이가 달라 최소 길이 "
                f"{min_seq} 토큰으로 잘라 평균을 계산합니다."
            )

        np.save(run_root / "global_layer_mean.npy", global_layer_mean)
        np.save(run_root / "global_mean.npy", global_mean)

        tokens_for_global = (
            reference_tokens[:min_seq]
            if reference_tokens is not None
            else list(range(min_seq))
        )
        plot_layer_grid(
            global_layer_mean,
            tokens_for_global,
            title_prefix="Global layer mean",
            output_path=run_root / "global_layer_mean.jpg",
        )
        plot_attention_map(
            global_mean,
            tokens_for_global,
            title="Global · Mean of all layers",
            output_path=run_root / "global_mean.jpg",
        )

        stats = {
            "run_name": run_name,
            "num_samples": len(samples),
            "layer_indices": layer_indices
            if layer_indices is not None
            else list(range(global_layer_mean.shape[0])),
            "aligned_seq_len": aligned_length,
            "per_sample": sample_stats,
        }
        with (run_root / "stats.json").open("w", encoding="utf-8") as file:
            json.dump(stats, file, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved visualization artifacts to: {run_root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
