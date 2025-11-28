"""Simple helpers for naming and saving attention bundles."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def default_run_name() -> str:
    """UTC timestamp used when the caller doesn't provide a run identifier."""

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def select_layers(
    attns: Sequence[Any], layers: Optional[Sequence[int]] = None
) -> Tuple[List[Any], List[int]]:
    """Return a filtered attention list plus the layer indices that were kept."""

    total_layers = len(attns)
    if layers is None:
        selected_indices = list(range(total_layers))
    else:
        selected_indices = sorted({int(idx) for idx in layers})
        for idx in selected_indices:
            if idx < 0 or idx >= total_layers:
                raise ValueError(
                    f"Layer index {idx} is out of range (0-{total_layers - 1})."
                )
    selected = [attns[idx] for idx in selected_indices]
    return selected, selected_indices


def save_attention_bundle(
    attn_list: Sequence[np.ndarray],
    tokens: Sequence[str],
    meta: Optional[Dict[str, Any]],
    out_dir: Union[str, Path],
) -> None:
    """Persist attention matrices, tokens, and metadata to disk."""

    if not attn_list:
        raise ValueError("attn_list must contain at least one layer.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    normalized: List[np.ndarray] = [np.asarray(arr, dtype=np.float32) for arr in attn_list]
    stacked = np.stack(normalized, axis=0)
    np.save(out_path / "attn.npy", stacked)

    tokens_list = list(tokens)
    with (out_path / "tokens.json").open("w", encoding="utf-8") as handle:
        json.dump(tokens_list, handle, ensure_ascii=False, indent=2)

    metadata = {
        "head_mean": True,
        "dtype": "float32",
        "seq_len": int(stacked.shape[-1]),
    }
    if meta:
        metadata.update(meta)
    metadata.setdefault("layers", list(range(stacked.shape[0])))

    with (out_path / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
