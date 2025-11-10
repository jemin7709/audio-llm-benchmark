from huggingface_hub import snapshot_download


def main() -> None:
    """Download MMAU-Pro's data.zip into the default HF cache.

    Environment variables like HF_HOME can be set by the user externally.
    """
    snapshot_path = snapshot_download(
        repo_id="gamma-lab-umd/MMAU-Pro", repo_type="dataset"
    )
    print(f"Snapshot path: {snapshot_path}")


if __name__ == "__main__":
    main()
