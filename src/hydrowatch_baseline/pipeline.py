from __future__ import annotations

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from scipy import ndimage


def _require_rasterio():
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as exc:
        raise RuntimeError("rasterio is required to run the baseline") from exc
    return rasterio, Resampling, reproject


def find_single(directory: Path, pattern: str, required: bool = True) -> Path | None:
    matches = [Path(path) for path in glob.glob(str(directory / pattern))]
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file matching {directory / pattern}, found {len(matches)}")
    return matches[0]


def read_s1(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    rasterio, _, _ = _require_rasterio()
    with rasterio.open(path) as source:
        if source.count < 2:
            raise ValueError(f"{path} must contain VV and VH as the first two bands")
        vv, vh = source.read([1, 2], out_dtype="float32", masked=True).filled(np.nan)
        profile = source.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    return vv, vh, profile


def read_optical_indices(path: Path | None, shape: tuple[int, int], config: dict):
    if path is None:
        missing = np.full(shape, np.nan, dtype=np.float32)
        return missing, missing.copy()
    rasterio, _, _ = _require_rasterio()
    bands = [config["optical"]["ndwi_band"], config["optical"]["mndwi_band"]]
    with rasterio.open(path) as source:
        if source.count < max(bands):
            raise ValueError(f"{path} has {source.count} bands but baseline needs band {max(bands)}")
        if (source.height, source.width) != shape:
            raise ValueError(f"Optical and SAR grids differ: {(source.height, source.width)} vs {shape}")
        ndwi, mndwi = source.read(bands, out_dtype="float32", masked=True).filled(np.nan)
    return ndwi, mndwi


def read_aux_on_grid(path: Path, target_profile: dict) -> np.ndarray:
    rasterio, Resampling, reproject = _require_rasterio()
    destination = np.empty((6, target_profile["height"], target_profile["width"]), dtype=np.float32)
    with rasterio.open(path) as source:
        if source.count < 6:
            raise ValueError(f"{path} must contain six auxiliary bands")
        for band in range(1, 7):
            reproject(
                source=rasterio.band(source, band), destination=destination[band - 1],
                src_transform=source.transform, src_crs=source.crs,
                dst_transform=target_profile["transform"], dst_crs=target_profile["crs"],
                resampling=Resampling.nearest if band == 6 else Resampling.bilinear,
            )
    return destination


def remove_small_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[labels]


def derive_masks(probabilities: np.ndarray, aux: np.ndarray, config: dict):
    model, hydrology = config["model"], config["hydrology"]
    slope, hand, occurrence, _, _, builtup = aux
    plausible = (
        np.isfinite(slope) & np.isfinite(hand)
        & (slope <= hydrology["slope_max_deg"])
        & (hand <= hydrology["hand_max_m"])
    )
    if hydrology["exclude_builtup"]:
        plausible &= np.nan_to_num(builtup, nan=0.0) < 0.5
    water_pre = (probabilities[..., 0] >= model["water_threshold"]) & plausible
    water_peak = (probabilities[..., 1] >= model["water_threshold"]) & plausible
    permanent = np.nan_to_num(occurrence, nan=0.0) >= hydrology["permanent_occurrence_min"]
    temporal_flood = water_peak & ~water_pre & ~permanent
    learned_flood = probabilities[..., 2] >= model["flood_threshold"]
    flood = temporal_flood & learned_flood
    if not flood.any() and temporal_flood.any():
        flood = temporal_flood
    return water_pre, water_peak, remove_small_components(
        flood, hydrology["minimum_component_pixels"]
    )


def pixel_area_ha(profile: dict) -> float:
    transform = profile["transform"]
    return abs(transform.a * transform.e - transform.b * transform.d) / 10_000.0


def write_mask(path: Path, mask: np.ndarray, profile: dict) -> None:
    rasterio, _, _ = _require_rasterio()
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(mask.astype("uint8"), 1)


def run_dataset(data_root: Path, output_dir: Path, model, config: dict) -> pd.DataFrame:
    from .sturm import build_eight_channel_input, predict_multimask

    pairs = pd.read_csv(data_root / "pairs.csv")
    rows = []
    sturm = config["sturm"]
    for pair in pairs.itertuples(index=False):
        directory = data_root / pair.rasters_dir
        vv_pre, vh_pre, profile = read_s1(find_single(directory, "S1_pre_*.tif"))
        vv_peak, vh_peak, peak_profile = read_s1(find_single(directory, "S1_peak_*.tif"))
        for key in ("width", "height", "crs", "transform"):
            if profile[key] != peak_profile[key]:
                raise ValueError(f"S1 grids differ for {pair.pair_id}: {key}")
        shape = vv_pre.shape
        ndwi_pre, mndwi_pre = read_optical_indices(
            find_single(directory, "SENTINEL2_pre_*.tif", required=False), shape, config
        )
        ndwi_peak, mndwi_peak = read_optical_indices(
            find_single(directory, "SENTINEL2_peak_*.tif", required=False), shape, config
        )
        model_input = build_eight_channel_input(
            vv_pre, vh_pre, vv_peak, vh_peak,
            ndwi_pre, mndwi_pre, ndwi_peak, mndwi_peak, config,
        )
        probabilities = predict_multimask(
            model_input, model, sturm["patch_size"], sturm["stride"], sturm["batch_size"]
        )
        aux = read_aux_on_grid(directory / "AUX_terrain_gsw.tif", profile)
        water_pre, water_peak, flood = derive_masks(probabilities, aux, config)
        prediction_dir = output_dir / "predictions"
        write_mask(prediction_dir / f"{pair.pair_id}_water_pre.tif", water_pre, profile)
        write_mask(prediction_dir / f"{pair.pair_id}_water_peak.tif", water_peak, profile)
        write_mask(prediction_dir / f"{pair.pair_id}_flood.tif", flood, profile)
        area = pixel_area_ha(profile)
        rows.append({
            "pair_id": pair.pair_id,
            "flood_ha": round(float(flood.sum() * area), 2),
            "water_pre_ha": round(float(water_pre.sum() * area), 2),
            "water_peak_ha": round(float(water_peak.sum() * area), 2),
        })
    submission = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    return submission
