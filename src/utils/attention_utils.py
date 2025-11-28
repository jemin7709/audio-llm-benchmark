"""Utility helpers for manipulating attention tensors."""

from __future__ import annotations

from typing import List

import numpy as np

__all__ = [
    "sanitize",
    "layer_list_to_array",
    "stack_batch",
    "mean_over_heads",
    "mean_over_layers",
]


def sanitize(array: np.ndarray) -> np.ndarray:
    """
    Ensure the given array is float32 and free of NaNs/Infs.
    """

    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def layer_list_to_array(layers: List[np.ndarray]) -> np.ndarray:
    """
    Convert a per-layer list of attention matrices into shape (L, S, S).
    """

    if not layers:
        raise ValueError("layers list must not be empty.")
    sanitized = [sanitize(layer) for layer in layers]
    return np.stack(sanitized, axis=0)


def stack_batch(batch: List[np.ndarray]) -> np.ndarray:
    """
    Stack multiple samples of shape (L, S, S) into (B, L, S, S).
    """

    if not batch:
        raise ValueError("batch must contain at least one sample.")
    for sample in batch:
        if sample.ndim != 3:
            raise ValueError(
                "Each sample must have shape (layers, seq_len, seq_len); "
                f"got ndim={sample.ndim}."
            )
    sanitized = [sanitize(sample) for sample in batch]
    return np.stack(sanitized, axis=0)


def mean_over_heads(attn: np.ndarray) -> np.ndarray:
    """
    Average the head dimension of a single-layer tensor: (H, S, S) -> (S, S).
    """

    if attn.ndim != 3:
        raise ValueError("Expected attention with ndim=3 (heads, seq, seq).")
    return sanitize(attn).mean(axis=0)


def mean_over_layers(attn_stack: np.ndarray) -> np.ndarray:
    """
    Average along the layer axis of either (L, S, S) or (B, L, S, S).
    """

    if attn_stack.ndim == 3:
        return sanitize(attn_stack).mean(axis=0)
    if attn_stack.ndim == 4:
        return sanitize(attn_stack).mean(axis=1)
    raise ValueError(
        "mean_over_layers expects a tensor with shape (L, S, S) or (B, L, S, S)."
    )
