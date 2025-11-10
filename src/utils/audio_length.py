from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import soundfile as sf


def iter_audio_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    """
    Yield audio file paths under root matching given extensions (case-insensitive).
    """
    lower_exts = {e.lower() for e in exts}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in lower_exts:
                yield Path(dirpath) / name


def get_duration_seconds(path: Path) -> float:
    """Return duration in seconds using header info without full decode."""
    with sf.SoundFile(path) as f:
        # duration = frames / samplerate
        return float(len(f)) / float(f.samplerate)


def compute_average_duration(audio_paths: Iterable[Path]) -> Tuple[float, int, float]:
    """
    Compute average duration (seconds), count, and total duration (seconds).
    Returns (avg_sec, count, total_sec).
    """
    total = 0.0
    count = 0
    for p in audio_paths:
        try:
            dur = get_duration_seconds(p)
        except Exception:
            # Skip unreadable files
            continue
        total += dur
        count += 1
    avg = (total / count) if count else 0.0
    return avg, count, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average audio duration under a directory"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=(
            "/home/jemin/.cache/huggingface/hub/datasets--woongvy--clotho-v2.1/"
            "snapshots/d3d9796c07aef72b521035874a31c1a8d35a06c5/validation"
        ),
        help="탐색할 디렉터리 절대경로",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".wav,.flac",
        help="대상 확장자 콤마구분 (기본: .wav,.flac)",
    )
    args = parser.parse_args()

    root = Path(args.dir).expanduser()
    extensions = [e.strip() for e in args.exts.split(",") if e.strip()]
    files: List[Path] = list(iter_audio_files(root, extensions))

    avg_sec, count, total_sec = compute_average_duration(files)

    print(f"Directory: {root}")
    print(f"Files counted: {count}")
    print(f"Total duration : {total_sec:.3f} seconds ({total_sec / 3600:.3f} hours)")
    print(f"Average duration (s): {avg_sec:.3f}")


if __name__ == "__main__":
    main()
