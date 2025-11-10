"""Utility: Reproducible seeding across common libs.

Minimal, framework-agnostic helper to set pseudo-random seeds for:
- Python `random`
- NumPy
- PyTorch (if installed)

This avoids repeating the same snippet in scripts.
"""

from __future__ import annotations

import os
import random


def seed_everything(
    seed: int = 42,
    *,
    deterministic: bool = True,
    numpy: bool = True,
    torch_cuda_all: bool = True,
) -> int:
    """Set seeds for Python, NumPy, and PyTorch (if available).

    Args:
        seed: Seed value.
        deterministic: If True and PyTorch is available, enable deterministic ops
            where possible (sets `torch.backends.cudnn.deterministic = True` and
            `torch.backends.cudnn.benchmark = False`).
        numpy: Whether to seed NumPy as well.
        torch_cuda_all: When CUDA is available, also call
            `torch.cuda.manual_seed_all(seed)`.

    Returns:
        The seed that was set (echoed for convenience).
    """

    # Core Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy (optional)
    if numpy:
        try:
            import numpy as _np  # type: ignore

            _np.random.seed(seed)
        except Exception:
            pass

    # PyTorch (optional)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass
        if (
            torch_cuda_all
            and getattr(torch, "cuda", None)
            and torch.cuda.is_available()
        ):
            try:
                torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        # PyTorch not installed or unavailable
        pass

    return seed


__all__ = ["seed_everything"]
