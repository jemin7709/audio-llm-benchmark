from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # 비-GUI 백엔드 강제 (멀티프로세스 안전)
import matplotlib.pyplot as plt
import numpy as np


# 한자 지원 폰트 설정
def _setup_cjk_font():
    """폰트 설정"""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans KR",
        "DejaVu Sans",
        "Arial",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


_setup_cjk_font()


def limited_tokens(tokens: Sequence[str]) -> Tuple[List[int], List[str]]:
    max_labels = 64
    total = len(tokens)
    if total <= max_labels:
        return list(range(total)), list(tokens)

    step = math.ceil(total / max_labels)
    indices = list(range(0, total, step))
    labels = [tokens[i] for i in indices]
    return indices, labels


def plot_attention_map(
    matrix: np.ndarray,
    tokens: Sequence[str],
    title: str,
    output_path: Path,
    scale: float = 1.0,
) -> None:
    data = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0) * float(scale)

    # 토큰 개수에 맞춰 이미지 크기 동적 조절 (토큰당 최소 0.25인치 확보)
    n_tokens = len(tokens)
    fig_size = max(10.0, n_tokens * 0.25)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    # interpolation='nearest'로 픽셀 경계 명확화
    cmap = ax.imshow(data, cmap="viridis", interpolation="nearest")
    fig.colorbar(cmap, ax=ax, fraction=0.046, pad=0.04)

    # 모든 토큰 표시 (생략 없음)
    ticks = list(range(n_tokens))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tokens, fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="both", which="both", length=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_layer_grid(
    layers: np.ndarray,
    tokens: Sequence[str],
    title_prefix: str,
    output_path: Path,
    scale: float = 1.0,
) -> None:
    count = layers.shape[0]
    cols = min(3, count)
    rows = math.ceil(count / cols)
    idx, labels = limited_tokens(tokens)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(rows, cols)

    for layer_idx in range(rows * cols):
        r, c = divmod(layer_idx, cols)
        ax = axes[r, c]
        if layer_idx >= count:
            ax.axis("off")
            continue
        data = np.nan_to_num(
            layers[layer_idx], nan=0.0, posinf=0.0, neginf=0.0
        ) * float(scale)
        im = ax.imshow(data, cmap="viridis")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(idx)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"{title_prefix} · L{layer_idx}")
        ax.tick_params(axis="both", which="both", length=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_sample_plots(
    attentions: Sequence[np.ndarray],
    tokens: Sequence[str],
    layer_indices: Sequence[int],
    sample_dir: Path,
    sample_id: str,
    img_format: str = "jpg",
    scale: float = 1.0,
) -> None:
    arrays = [np.asarray(attn, dtype=np.float32) for attn in attentions]
    stacked = np.stack(arrays, axis=0)

    for layer_idx, matrix in zip(layer_indices, stacked):
        plot_attention_map(
            matrix,
            tokens,
            title=f"{sample_id} · Layer {layer_idx}",
            output_path=sample_dir / f"layer_{layer_idx:02d}.{img_format}",
            scale=scale,
        )

    mean_matrix = stacked.mean(axis=0)
    plot_attention_map(
        mean_matrix,
        tokens,
        title=f"{sample_id} · Layer mean",
        output_path=sample_dir / f"layer_mean.{img_format}",
        scale=scale,
    )


def render_sample_from_disk(
    sample_dir: Path,
    img_format: str = "jpg",
    overwrite: bool = False,
    scale: float = 1.0,
) -> None:
    """Load previously saved attention bundle and materialize plot files."""

    sample_dir = Path(sample_dir)
    attn_path = sample_dir / "attn.npy"
    tokens_path = sample_dir / "tokens.json"
    meta_path = sample_dir / "meta.json"

    if not attn_path.exists():
        raise FileNotFoundError(f"Missing attention array: {attn_path}")
    if not tokens_path.exists():
        raise FileNotFoundError(f"Missing tokens file: {tokens_path}")

    stacked = np.load(attn_path)
    with tokens_path.open("r", encoding="utf-8") as handle:
        tokens: Sequence[str] = json.load(handle)

    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

    layer_indices = meta.get("layers") or list(range(stacked.shape[0]))
    if len(layer_indices) != stacked.shape[0]:
        layer_indices = list(range(stacked.shape[0]))
    sample_id = meta.get("sample_id") or sample_dir.name

    for local_idx, layer_number in enumerate(layer_indices):
        out_file = sample_dir / f"layer_{int(layer_number):02d}.{img_format}"
        if out_file.exists() and not overwrite:
            continue
        plot_attention_map(
            stacked[local_idx],
            tokens,
            title=f"{sample_id} · Layer {layer_number}",
            output_path=out_file,
            scale=scale,
        )

    mean_path = sample_dir / f"layer_mean.{img_format}"
    if overwrite or not mean_path.exists():
        plot_attention_map(
            stacked.mean(axis=0),
            tokens,
            title=f"{sample_id} · Layer mean",
            output_path=mean_path,
            scale=scale,
        )


def _render_sample_worker(
    args: Tuple[Path, str, bool, float],
) -> Tuple[str, str | None]:
    """Worker function for parallel rendering."""
    sample_dir, img_format, overwrite, scale = args
    try:
        render_sample_from_disk(
            sample_dir, img_format=img_format, overwrite=overwrite, scale=scale
        )
        return (sample_dir.name, None)
    except FileNotFoundError as exc:
        return (sample_dir.name, str(exc))


def render_attention_run(
    root: Path,
    img_format: str = "jpg",
    overwrite: bool = False,
    workers: int | None = None,
    scale: float = 1.0,
) -> None:
    """Render plots for every sample_* directory inside the provided root path."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Attention root not found: {root}")

    sample_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories under {root}")

    # attn.npy 있는 디렉토리만 필터
    valid_dirs = [d for d in sample_dirs if (d / "attn.npy").exists()]
    if not valid_dirs:
        print(f"[WARN] No valid sample directories with attn.npy under {root}")
        return

    max_workers = workers or min(os.cpu_count() or 4, len(valid_dirs))
    tasks = [(d, img_format, overwrite, scale) for d in valid_dirs]

    print(f"[INFO] Rendering {len(valid_dirs)} samples with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_render_sample_worker, t): t[0].name for t in tasks}
        done_count = 0
        for future in as_completed(futures):
            name, err = future.result()
            done_count += 1
            if err:
                print(f"[WARN] Skipping {name}: {err}")
            else:
                print(f"[{done_count}/{len(valid_dirs)}] Done: {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render saved attention bundles into image files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory that contains sample_* subfolders with attn.npy files.",
    )
    parser.add_argument(
        "--img-format",
        default="png",
        help="Image format/extension to emit (default: png).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing attention images.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Intensity scale factor applied to attention values before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    render_attention_run(
        args.root,
        img_format=args.img_format,
        overwrite=args.overwrite,
        workers=args.workers,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
