"""Download KuaiRand-Pure without redistributing the dataset in this repository."""
from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1"
EXPECTED_MD5 = "0820331067a3784d9691136f772b35a7"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "KuaiRand-Pure.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: {digest}; expected {EXPECTED_MD5}")
    with tarfile.open(archive) as source:
        source.extractall(args.output, filter="data")
    print(f"Downloaded and verified {archive}")


if __name__ == "__main__":
    main()
