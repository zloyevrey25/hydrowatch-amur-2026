from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio


def find_one(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {pattern} in {directory}, found {len(paths)}")
    return paths[0]


def water_mask(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        vv = src.read(1)
        vh = src.read(2)
        profile = src.profile.copy()
        valid = np.isfinite(vv) & np.isfinite(vh)
        if src.nodata is not None:
            valid &= (vv != src.nodata) & (vh != src.nodata)
    # Conservative Sentinel-1 water threshold in dB. Requiring both
    # polarisations limits dark soil and radar-shadow false positives.
    water = valid & (vv < -16.0) & (vh < -23.0)
    return water, profile


def write_mask(path: Path, mask: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def main() -> None:
    parser = ArgumentParser(description="Fast deterministic SAR fallback submission")
    parser.add_argument("--data-root", type=Path, default=Path("data/hydrowatch_amur"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final"))
    args = parser.parse_args()

    pairs = pd.read_csv(args.data_root / "pairs.csv")
    prediction_dir = args.output_dir / "predictions"
    rows: list[dict] = []
    for pair in pairs.itertuples(index=False):
        directory = args.data_root / pair.rasters_dir
        pre, profile = water_mask(find_one(directory, "S1_pre_*.tif"))
        peak, peak_profile = water_mask(find_one(directory, "S1_peak_*.tif"))
        for key in ("width", "height", "crs", "transform"):
            if profile[key] != peak_profile[key]:
                raise RuntimeError(f"Grid mismatch for {pair.pair_id}: {key}")
        flood = peak & ~pre
        if pair.event_kind == "baseline":
            flood[:] = False
        receded = pre & ~peak
        write_mask(prediction_dir / f"{pair.pair_id}_water_pre.tif", pre, profile)
        write_mask(prediction_dir / f"{pair.pair_id}_water_peak.tif", peak, profile)
        write_mask(prediction_dir / f"{pair.pair_id}_flood.tif", flood, profile)
        write_mask(prediction_dir / f"{pair.pair_id}_receded.tif", receded, profile)
        pixel_ha = abs(
            profile["transform"].a * profile["transform"].e
            - profile["transform"].b * profile["transform"].d
        ) / 10_000.0
        rows.append({
            "pair_id": pair.pair_id,
            "flood_ha": round(float(flood.sum()) * pixel_ha, 2),
            "water_pre_ha": round(float(pre.sum()) * pixel_ha, 2),
            "water_peak_ha": round(float(peak.sum()) * pixel_ha, 2),
        })
        print(f"finished {pair.pair_id}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "submission.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "method": "conservative_sentinel1_threshold_fallback",
        "vv_threshold_db": -16.0,
        "vh_threshold_db": -23.0,
        "pairs": [row["pair_id"] for row in rows],
        "note": "Emergency deterministic fallback after Colab CUDA autotuning stalled.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
