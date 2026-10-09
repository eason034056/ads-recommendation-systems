"""Download Criteo's public attribution dataset without redistributing it."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/criteo_attribution_dataset.tsv.gz?download=true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "criteo_attribution_dataset.tsv.gz"
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    if path.stat().st_size < 500_000_000:
        raise RuntimeError("Unexpectedly small Criteo download; delete the file and retry")
    print(f"Downloaded {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
