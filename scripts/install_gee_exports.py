from __future__ import annotations

from pathlib import Path
import argparse
import shutil

import pandas as pd
import rasterio


def same_grid(reference: Path, candidate: Path) -> tuple[bool, str]:
    with rasterio.open(reference) as expected, rasterio.open(candidate) as actual:
        checks = {
            "width": (expected.width, actual.width),
            "height": (expected.height, actual.height),
            "crs": (expected.crs, actual.crs),
            "transform": (expected.transform, actual.transform),
        }
    differences = [f"{name}: {left} != {right}" for name, (left, right) in checks.items() if left != right]
    return not differences, "; ".join(differences)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install downloaded Earth Engine exports into pair directories")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pairs = pd.read_csv(args.data_root / "pairs.csv").set_index("pair_id")
    installed = 0
    rejected: list[str] = []
    for source in sorted(args.source_dir.glob("*.tif")):
        matches = [pair_id for pair_id in pairs.index if source.name.startswith(f"{pair_id}__")]
        if len(matches) != 1:
            rejected.append(f"{source.name}: cannot determine pair")
            continue
        pair_id = matches[0]
        filename = source.name[len(pair_id) + 2 :]
        destination_dir = args.data_root / pairs.loc[pair_id, "rasters_dir"]
        destination = destination_dir / filename
        reference = args.data_root / pairs.loc[pair_id, "reference_mask"]
        valid, reason = same_grid(reference, source)
        if not valid:
            rejected.append(f"{source.name}: grid mismatch: {reason}")
            continue
        if destination.exists() and not args.overwrite:
            rejected.append(f"{source.name}: destination exists")
            continue
        destination_dir.mkdir(parents=True, exist_ok=True)
        if args.move:
            shutil.move(source, destination)
        else:
            shutil.copy2(source, destination)
        installed += 1
        print(f"Installed {destination}")

    print(f"Installed files: {installed}")
    if rejected:
        print("Skipped files:")
        for message in rejected:
            print(f"- {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
