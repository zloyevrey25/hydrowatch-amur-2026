from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import tarfile
import tempfile
import urllib.request


URL = "https://zenodo.org/records/15189665/files/S1_model.tar.gz"
EXPECTED_MD5 = "14a046d9d7965f2a3c511acb1bbca57b"


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Unsupported archive member: {member.name}")
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe archive member: {member.name}")
    # Python 3.9 has no tarfile extraction_filter argument. The checks above
    # provide the equivalent guarantees needed for this known model archive.
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public STURM-Flood Sentinel-1 weights")
    parser.add_argument("--destination", type=Path, default=Path("models/sturm_s1"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as temporary:
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL) as response:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
        temporary.flush()
        digest = hashlib.md5(usedforsecurity=False)
        with Path(temporary.name).open("rb") as downloaded:
            while chunk := downloaded.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != EXPECTED_MD5:
            raise ValueError(
                f"Checksum mismatch: expected {EXPECTED_MD5}, got {digest.hexdigest()}"
            )
        print(f"Verified MD5: {EXPECTED_MD5}")
        with tarfile.open(temporary.name, "r:gz") as archive:
            safe_extract(archive, args.destination)
    print(f"Weights extracted to {args.destination}")


if __name__ == "__main__":
    main()
