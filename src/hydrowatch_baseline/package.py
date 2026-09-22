from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("pair_id", "flood_ha", "water_pre_ha", "water_peak_ha")


@dataclass(frozen=True)
class PackageValidation:
    valid: bool
    pairs: int
    masks: int
    errors: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _same_grid(reference, mask) -> list[str]:
    differences = []
    for name in ("width", "height", "crs", "transform"):
        if getattr(reference, name) != getattr(mask, name):
            differences.append(name)
    return differences


def validate_submission_package(
    data_root: str | Path,
    package_dir: str | Path,
    *,
    area_tolerance: float = 0.02,
) -> PackageValidation:
    """Validate the CSV and mandatory flood rasters as one submission package."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required to validate a submission package") from exc

    data_root, package_dir = Path(data_root), Path(package_dir)
    submission_path = package_dir / "submission.csv"
    prediction_dir = package_dir / "predictions"
    pairs = pd.read_csv(data_root / "pairs.csv")
    errors: list[str] = []

    if not submission_path.exists():
        return PackageValidation(False, len(pairs), 0, ["submission.csv is missing"])

    submission = pd.read_csv(submission_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(submission.columns))
    if missing_columns:
        return PackageValidation(
            False, len(pairs), 0, [f"submission.csv is missing columns: {missing_columns}"]
        )

    expected_ids = list(pairs["pair_id"])
    actual_ids = list(submission["pair_id"])
    duplicates = sorted(submission.loc[submission["pair_id"].duplicated(), "pair_id"].unique())
    missing_ids = sorted(set(expected_ids) - set(actual_ids))
    unknown_ids = sorted(set(actual_ids) - set(expected_ids))
    if duplicates:
        errors.append(f"duplicate pair_id values: {duplicates}")
    if missing_ids:
        errors.append(f"missing pair_id values: {missing_ids}")
    if unknown_ids:
        errors.append(f"unknown pair_id values: {unknown_ids}")

    areas = submission[list(REQUIRED_COLUMNS[1:])].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(areas.to_numpy()).all():
        errors.append("area values contain NaN or infinity")
    if (areas < 0).any().any():
        errors.append("area values must be non-negative")
    if (areas["flood_ha"] > areas["water_peak_ha"]).any():
        bad = submission.loc[areas["flood_ha"] > areas["water_peak_ha"], "pair_id"].tolist()
        errors.append(f"flood_ha exceeds water_peak_ha: {bad}")

    joined = pairs[["pair_id", "aoi_km2", "reference_mask"]].merge(
        submission[list(REQUIRED_COLUMNS)], on="pair_id", how="left", validate="one_to_many"
    )
    for column in REQUIRED_COLUMNS[1:]:
        too_large = pd.to_numeric(joined[column], errors="coerce") > joined["aoi_km2"] * 100.0
        if too_large.any():
            errors.append(f"{column} exceeds AOI area: {joined.loc[too_large, 'pair_id'].tolist()}")

    masks_found = 0
    for pair in pairs.itertuples(index=False):
        mask_path = prediction_dir / f"{pair.pair_id}_flood.tif"
        if not mask_path.exists():
            errors.append(f"missing mask: predictions/{pair.pair_id}_flood.tif")
            continue
        masks_found += 1
        try:
            with rasterio.open(data_root / pair.reference_mask) as reference, rasterio.open(mask_path) as mask:
                grid_differences = _same_grid(reference, mask)
                if grid_differences:
                    errors.append(f"{mask_path.name}: grid mismatch in {grid_differences}")
                    continue
                if mask.count != 1:
                    errors.append(f"{mask_path.name}: expected 1 band, got {mask.count}")
                    continue
                if mask.dtypes[0] != "uint8":
                    errors.append(f"{mask_path.name}: expected uint8, got {mask.dtypes[0]}")
                values = mask.read(1)
                unexpected = np.setdiff1d(np.unique(values), np.array([0, 1], dtype=values.dtype))
                if unexpected.size:
                    errors.append(f"{mask_path.name}: non-binary values {unexpected.tolist()}")
                    continue
                pixel_ha = abs(
                    mask.transform.a * mask.transform.e - mask.transform.b * mask.transform.d
                ) / 10_000.0
                mask_area = float(values.sum()) * pixel_ha
        except Exception as exc:
            errors.append(f"{mask_path.name}: cannot read raster: {exc}")
            continue

        csv_rows = submission.loc[submission["pair_id"] == pair.pair_id, "flood_ha"]
        if len(csv_rows) != 1 or not np.isfinite(pd.to_numeric(csv_rows, errors="coerce").iloc[0]):
            continue
        csv_area = float(csv_rows.iloc[0])
        allowed = max(0.01, pixel_ha, area_tolerance * max(csv_area, mask_area))
        if abs(csv_area - mask_area) > allowed:
            errors.append(
                f"{mask_path.name}: mask area {mask_area:.2f} ha differs from CSV "
                f"{csv_area:.2f} ha by more than 2%"
            )

    return PackageValidation(not errors, len(pairs), masks_found, errors)


def write_package_validation(result: PackageValidation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
