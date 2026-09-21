from __future__ import annotations

from pathlib import Path
import glob
import json

import numpy as np
import pandas as pd


REFERENCE_CHANNELS = ("flood", "water_pre", "water_peak", "permanent", "receded")
MODEL_TARGET_CHANNELS = ("water_pre", "water_peak", "flood")


def _rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required for dataset preparation") from exc
    return rasterio


def _one(directory: Path, pattern: str) -> Path | None:
    matches = sorted(Path(path) for path in glob.glob(str(directory / pattern)))
    return matches[0] if len(matches) == 1 else None


def audit_dataset(data_root: Path) -> pd.DataFrame:
    """Inspect every supplied pair without reading unavailable Sentinel scenes."""
    rasterio = _rasterio()
    pairs = pd.read_csv(data_root / "pairs.csv")
    rows: list[dict] = []
    for pair in pairs.itertuples(index=False):
        pair_dir = data_root / pair.rasters_dir
        reference = data_root / pair.reference_mask
        problems: list[str] = []
        areas = {f"{name}_ha": np.nan for name in REFERENCE_CHANNELS}
        height = width = band_count = 0
        crs = ""
        resolution_m = np.nan
        if not reference.exists():
            problems.append("reference_missing")
        else:
            with rasterio.open(reference) as source:
                height, width, band_count = source.height, source.width, source.count
                crs = str(source.crs)
                resolution_m = float(abs(source.transform.a))
                if source.count != len(REFERENCE_CHANNELS):
                    problems.append(f"reference_bands={source.count}")
                masks = source.read(out_dtype="uint8")
                if np.count_nonzero(~np.isin(masks, (0, 1))):
                    problems.append("reference_non_binary")
                pixel_ha = abs(
                    source.transform.a * source.transform.e
                    - source.transform.b * source.transform.d
                ) / 10_000.0
                for index, name in enumerate(REFERENCE_CHANNELS[: source.count]):
                    areas[f"{name}_ha"] = round(float(masks[index].sum() * pixel_ha), 2)
                if str(source.crs) != "EPSG:32652":
                    problems.append(f"reference_crs={source.crs}")
                if not np.isclose(resolution_m, 10.0):
                    problems.append(f"reference_resolution={resolution_m}")

        aux = pair_dir / "AUX_terrain_gsw.tif"
        if not aux.exists():
            problems.append("aux_missing")
        else:
            with rasterio.open(aux) as source:
                if source.count != 6:
                    problems.append(f"aux_bands={source.count}")

        s1_pre = _one(pair_dir, "S1_pre_*.tif")
        s1_peak = _one(pair_dir, "S1_peak_*.tif")
        s2_pre = _one(pair_dir, "SENTINEL2_pre_*.tif")
        s2_peak = _one(pair_dir, "SENTINEL2_peak_*.tif")
        if s1_pre is None:
            problems.append("s1_pre_missing")
        if s1_peak is None:
            problems.append("s1_peak_missing")
        metadata_has_optical = pd.notna(getattr(pair, "sensor_optical", np.nan))
        optical_ready = s2_pre is not None and s2_peak is not None
        if metadata_has_optical and not optical_ready:
            problems.append("s2_tiffs_missing")

        rows.append({
            "pair_id": pair.pair_id,
            "event_id": pair.event_id,
            "event_kind": pair.event_kind,
            "aoi_id": pair.aoi_id,
            "height": height,
            "width": width,
            "reference_bands": band_count,
            "crs": crs,
            "resolution_m": resolution_m,
            "metadata_has_optical": metadata_has_optical,
            "s1_ready": s1_pre is not None and s1_peak is not None,
            "s2_ready": optical_ready,
            **areas,
            "status": "ready" if not problems else "incomplete",
            "problems": ";".join(problems),
        })
    return pd.DataFrame(rows)


def build_event_folds(data_root: Path) -> pd.DataFrame:
    """Assign one fold per whole hydrological event to prevent spatial leakage."""
    pairs = pd.read_csv(data_root / "pairs.csv")
    events = sorted(pairs["event_id"].unique())
    event_to_fold = {event: fold for fold, event in enumerate(events)}
    result = pairs[["pair_id", "event_id", "event_kind", "aoi_id"]].copy()
    result["fold"] = result["event_id"].map(event_to_fold)
    return result


def build_patch_manifest(
    data_root: Path,
    patch_size: int = 128,
    stride: int = 128,
) -> pd.DataFrame:
    """Index reference patches and their class balance without loading imagery."""
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    rasterio = _rasterio()
    from rasterio.windows import Window

    pairs = pd.read_csv(data_root / "pairs.csv")
    folds = build_event_folds(data_root).set_index("pair_id")
    rows: list[dict] = []

    def starts(length: int) -> list[int]:
        if length < patch_size:
            raise ValueError(f"Raster dimension {length} is smaller than patch size {patch_size}")
        values = list(range(0, length - patch_size + 1, stride))
        if values[-1] != length - patch_size:
            values.append(length - patch_size)
        return values

    for pair in pairs.itertuples(index=False):
        reference = data_root / pair.reference_mask
        with rasterio.open(reference) as source:
            for row in starts(source.height):
                for col in starts(source.width):
                    labels = source.read(
                        [1, 2, 3], window=Window(col, row, patch_size, patch_size),
                        out_dtype="uint8",
                    )
                    rows.append({
                        "pair_id": pair.pair_id,
                        "event_id": pair.event_id,
                        "event_kind": pair.event_kind,
                        "aoi_id": pair.aoi_id,
                        "fold": int(folds.loc[pair.pair_id, "fold"]),
                        "row": row,
                        "col": col,
                        "patch_size": patch_size,
                        "flood_fraction": round(float(labels[0].mean()), 8),
                        "water_pre_fraction": round(float(labels[1].mean()), 8),
                        "water_peak_fraction": round(float(labels[2].mean()), 8),
                    })
    return pd.DataFrame(rows)


def write_preparation_manifests(
    data_root: Path,
    output_dir: Path,
    patch_size: int = 128,
    stride: int = 128,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "audit": output_dir / "data_audit.csv",
        "folds": output_dir / "event_folds.csv",
        "patches": output_dir / "patch_manifest.csv",
        "summary": output_dir / "dataset_summary.json",
    }
    audit = audit_dataset(data_root)
    folds = build_event_folds(data_root)
    patches = build_patch_manifest(data_root, patch_size, stride)
    audit.to_csv(outputs["audit"], index=False)
    folds.to_csv(outputs["folds"], index=False)
    patches.to_csv(outputs["patches"], index=False)
    event_pairs = audit[audit["event_kind"] != "baseline"]
    controls = audit[audit["event_kind"] == "baseline"]
    summary = {
        "pairs": int(len(audit)),
        "events": int(audit["event_id"].nunique()),
        "event_pairs": int(len(event_pairs)),
        "control_pairs": int(len(controls)),
        "metadata_optical_pairs": int(audit["metadata_has_optical"].sum()),
        "ready_s1_pairs": int(audit["s1_ready"].sum()),
        "ready_s2_pairs": int(audit["s2_ready"].sum()),
        "patches": int(len(patches)),
        "flood_positive_patches": int((patches["flood_fraction"] > 0).sum()),
        "reference_event_flood_ha": {
            "min": round(float(event_pairs["flood_ha"].min()), 2),
            "median": round(float(event_pairs["flood_ha"].median()), 2),
            "max": round(float(event_pairs["flood_ha"].max()), 2),
        },
        "control_reference_flood_ha": {
            row.pair_id: float(row.flood_ha) for row in controls.itertuples(index=False)
        },
    }
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs
