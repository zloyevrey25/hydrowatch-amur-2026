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
    matches = [Path(p) for p in glob.glob(str(directory / pattern))]
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file matching {directory / pattern}, found {len(matches)}")
    return matches[0]


def read_s1(path: Path, config: dict) -> tuple[np.ndarray, dict]:
    from .sturm import normalize_s1

    rasterio, _, _ = _require_rasterio()
    with rasterio.open(path) as source:
        if source.count < 2:
            raise ValueError(f"{path} must contain VV and VH as the first two bands")
        vv, vh = source.read([1, 2], out_dtype="float32")
        profile = source.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    return normalize_s1(vv, vh, config), profile


def optical_probability(path: Path | None, config: dict) -> np.ndarray | None:
    if path is None:
        return None
    rasterio, _, _ = _require_rasterio()
    optical = config["optical"]
    indices = [
        optical["ndwi_band"], optical["mndwi_band"],
        optical["ndvi_band"], optical["awei_band"],
    ]
    with rasterio.open(path) as source:
        if source.count < max(indices):
            raise ValueError(f"{path} has {source.count} bands but baseline needs band {max(indices)}")
        ndwi, mndwi, ndvi, awei = source.read(indices, out_dtype="float32")
    valid = np.isfinite(ndwi) & np.isfinite(mndwi) & np.isfinite(ndvi) & np.isfinite(awei)
    water = (
        (ndwi > optical["ndwi_threshold"])
        & (mndwi > optical["mndwi_threshold"])
        & (ndvi <= optical["ndvi_max"])
        & (awei > optical["awei_threshold"])
    )
    result = np.full(water.shape, np.nan, dtype=np.float32)
    result[valid] = water[valid].astype(np.float32)
    return result


def fuse_probability(sar: np.ndarray, optical: np.ndarray | None, config: dict) -> np.ndarray:
    if optical is None:
        return sar
    if optical.shape != sar.shape:
        raise ValueError(f"Optical and SAR grids differ: {optical.shape} vs {sar.shape}")
    fusion = config["fusion"]
    valid = np.isfinite(optical)
    result = sar.copy()
    result[valid] = (
        fusion["sar_weight_with_optical"] * sar[valid]
        + fusion["optical_weight"] * optical[valid]
    )
    return result


def read_aux_on_grid(path: Path, target_profile: dict) -> np.ndarray:
    rasterio, Resampling, reproject = _require_rasterio()
    destination = np.empty((6, target_profile["height"], target_profile["width"]), dtype=np.float32)
    with rasterio.open(path) as source:
        if source.count < 6:
            raise ValueError(f"{path} must contain six auxiliary bands")
        for band in range(1, 7):
            reproject(
                source=rasterio.band(source, band),
                destination=destination[band - 1],
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=target_profile["transform"],
                dst_crs=target_profile["crs"],
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


def derive_masks(pre_probability: np.ndarray, peak_probability: np.ndarray, aux: np.ndarray, config: dict):
    threshold = config["fusion"]["water_threshold"]
    hydrology = config["hydrology"]
    slope, hand, occurrence, _, _, builtup = aux
    plausible = (
        np.isfinite(slope)
        & np.isfinite(hand)
        & (slope <= hydrology["slope_max_deg"])
        & (hand <= hydrology["hand_max_m"])
    )
    if hydrology["exclude_builtup"]:
        plausible &= np.nan_to_num(builtup, nan=0.0) < 0.5

    water_pre = (pre_probability >= threshold) & plausible
    water_peak = (peak_probability >= threshold) & plausible
    permanent = np.nan_to_num(occurrence, nan=0.0) >= hydrology["permanent_occurrence_min"]
    flood = water_peak & ~water_pre & ~permanent
    flood = remove_small_components(flood, hydrology["minimum_component_pixels"])
    return water_pre, water_peak, flood


def pixel_area_ha(profile: dict) -> float:
    transform = profile["transform"]
    area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    return area_m2 / 10_000.0


def write_mask(path: Path, mask: np.ndarray, profile: dict) -> None:
    rasterio, _, _ = _require_rasterio()
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(mask.astype("uint8"), 1)


def run_dataset(data_root: Path, output_dir: Path, model, config: dict) -> pd.DataFrame:
    from .sturm import predict_probability

    pairs = pd.read_csv(data_root / "pairs.csv")
    rows = []
    sturm = config["sturm"]
    for pair in pairs.itertuples(index=False):
        directory = data_root / pair.rasters_dir
        pre_path = find_single(directory, "S1_pre_*.tif")
        peak_path = find_single(directory, "S1_peak_*.tif")
        opt_pre_path = find_single(directory, "SENTINEL2_pre_*.tif", required=False)
        opt_peak_path = find_single(directory, "SENTINEL2_peak_*.tif", required=False)
        aux_path = directory / "AUX_terrain_gsw.tif"

        s1_pre, profile = read_s1(pre_path, config)
        s1_peak, peak_profile = read_s1(peak_path, config)
        for key in ("width", "height", "crs", "transform"):
            if profile[key] != peak_profile[key]:
                raise ValueError(f"S1 grids differ for {pair.pair_id}: {key}")

        pre_probability = predict_probability(s1_pre, model, sturm["patch_size"], sturm["stride"], sturm["batch_size"], sturm["water_class_index"])
        peak_probability = predict_probability(s1_peak, model, sturm["patch_size"], sturm["stride"], sturm["batch_size"], sturm["water_class_index"])
        pre_probability = fuse_probability(pre_probability, optical_probability(opt_pre_path, config), config)
        peak_probability = fuse_probability(peak_probability, optical_probability(opt_peak_path, config), config)

        aux = read_aux_on_grid(aux_path, profile)
        water_pre, water_peak, flood = derive_masks(pre_probability, peak_probability, aux, config)
        write_mask(output_dir / "predictions" / f"{pair.pair_id}_flood.tif", flood, profile)
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

