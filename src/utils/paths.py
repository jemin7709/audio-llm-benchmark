"""Path utilities for project root detection and standard directory management."""

import os
import typing as t
from pathlib import Path


def detect_repo_root(*, env_var: str = "LALM_BENCH_ROOT") -> Path:
    """
    Detect project root directory.

    Tries in order:
    1. Environment variable LALM_BENCH_ROOT (if set)
    2. Current working directory (if contains 'envs/')
    3. Parent directories of __file__ until 'envs/' is found

    Args:
        env_var: Environment variable name to check (default: LALM_BENCH_ROOT)

    Returns:
        Path to project root

    Raises:
        RuntimeError: If root cannot be detected
    """
    candidates: list[Path] = []

    # 1. Check environment variable
    env_override = os.environ.get(env_var)
    if env_override:
        candidates.append(Path(env_override).expanduser())

    # 2. Check current working directory
    candidates.append(Path.cwd())

    # 3. Check module location and parents
    candidates.append(Path(__file__).resolve().parent.parent.parent)

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "envs").exists():
            return resolved

    raise RuntimeError(
        "프로젝트 루트를 찾을 수 없습니다. "
        "프로젝트 루트에서 명령을 실행하거나 LALM_BENCH_ROOT를 설정하세요."
    )


def resolve_under_root(path: t.Union[Path, str], root: Path) -> Path:
    """
    Resolve path relative to root if it's relative; return as-is if absolute.

    Args:
        path: Path to resolve (can be str or Path)
        root: Project root directory

    Returns:
        Absolute Path (either the original if absolute, or root/path)
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def project_paths(root: Path) -> dict[str, Path]:
    """
    Return standard project directories.

    Args:
        root: Project root directory

    Returns:
        Dictionary with keys: data_dir, outputs_dir, temp_dir
    """
    return {
        "data_dir": root / "data",
        "outputs_dir": root / "outputs",
        "temp_dir": root / "temp",
    }


def resolve_repo_file(rel: str, root: Path) -> Path:
    """
    Resolve a file relative to repo root (regardless of cwd).

    Args:
        rel: Relative path from repo root (e.g., "white-noise-358382.mp3")
        root: Project root directory

    Returns:
        Absolute path to the file
    """
    return (root / rel).resolve()


def ensure_dir(path: Path) -> Path:
    """
    Create directory and all parents; return path.

    Args:
        path: Directory path to ensure exists

    Returns:
        The same path (now guaranteed to exist)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_output_path(path: t.Union[Path, str], root: Path) -> Path:
    """
    Normalize output path relative to root if relative; absolute otherwise.

    Args:
        path: Output path (can be str or Path)
        root: Project root directory

    Returns:
        Absolute Path (either the original if absolute, or root/path)
    """
    return resolve_under_root(path, root)

