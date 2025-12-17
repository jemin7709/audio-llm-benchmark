import argparse
import json
from pathlib import Path
from tqdm import tqdm


def read_last_object_from_json_list(file_path: Path) -> dict:
    """Load a JSON file expected to contain a list and return the last object.

    Raises:
        ValueError: If the JSON is not a list or is empty.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON is not a list: {file_path}")
    if not data:
        raise ValueError(f"JSON list is empty: {file_path}")

    last_object = data[-1]
    if not isinstance(last_object, dict):
        raise ValueError(f"Last element is not a JSON object: {file_path}")
    return last_object


def write_json(content: object, dst_path: Path, wrap_list: bool) -> None:
    """Write content as JSON to dst_path. Optionally wrap in a list with one element."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [content] if wrap_list else content
    # Compact JSON to reduce file size; preserve UTF-8 characters
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(dst_path)


def process_directory(
    src_dir: Path,
    dst_dir: Path,
    pattern: str,
    wrap_list: bool,
    overwrite: bool,
) -> None:
    json_files = sorted(src_dir.glob(pattern))
    if not json_files:
        print(f"No files matched: {src_dir}/{pattern}")
        return

    total = len(json_files)
    processed = 0
    skipped = 0
    failed = 0

    for src_path in tqdm(json_files, desc="Processing JSON files", unit="file"):
        if not src_path.is_file() or src_path.suffix.lower() != ".json":
            continue
        rel_name = src_path.name
        dst_path = dst_dir / rel_name

        if dst_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            last_obj = read_last_object_from_json_list(src_path)
            write_json(last_obj, dst_path, wrap_list=wrap_list)
            processed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {src_path}: {e}")

    print(
        f"Done. processed={processed}, skipped={skipped}, failed={failed}, total={total}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load each JSON file from a directory containing a list, keep only the last "
            "JSON object, and write to a mirror directory with the same filenames."
        )
    )
    default_src = Path(__file__).resolve().parent / "stoch"
    default_dst = Path(__file__).resolve().parent / "stoch_new"
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=default_src,
        help=f"Source directory containing JSON files (default: {default_src})",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=default_dst,
        help=f"Destination directory (default: {default_dst})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help='Glob pattern for source files (default: "*.json")',
    )
    parser.add_argument(
        "--wrap-list",
        action="store_true",
        help="Write output as a one-element list instead of a single object",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in destination directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir: Path = args.src_dir
    dst_dir: Path = args.dst_dir
    pattern: str = args.pattern
    wrap_list: bool = args.wrap_list
    overwrite: bool = args.overwrite

    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Source directory not found: {src_dir}")

    process_directory(
        src_dir=src_dir,
        dst_dir=dst_dir,
        pattern=pattern,
        wrap_list=wrap_list,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
