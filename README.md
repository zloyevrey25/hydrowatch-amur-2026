# HydroWatch Amur baseline

This project adapts the public STURM-Flood Sentinel-1 U-Net into an eight-channel
multimodal baseline for the CosmoHackathon 2026 hydrological monitoring case.

The baseline performs the following steps:

1. reads VV and VH sigma0 bands for the `pre` and `peak` dates;
2. reads NDWI and MNDWI from Sentinel-2 for both dates;
3. builds one eight-channel tensor in this fixed order:
   `VV_pre, VH_pre, VV_peak, VH_peak, NDWI_pre, MNDWI_pre, NDWI_peak, MNDWI_peak`;
4. runs overlapping 128 x 128 multimodal U-Net inference;
5. removes implausible pixels using slope, HAND, permanent water and built-up layers;
6. predicts `water_pre`, `water_peak` and `flood`, then constrains the flood mask
   with the temporal change and permanent-water mask;
7. writes three GeoTIFF masks per pair and `submission.csv`;
8. computes the local competition Score and all four components.

## Expected data

Extract the case archive so that `pairs.csv`, `reference_masks/`, `vectors/` and
`rasters/` are under one data root. The lite archive does not contain Sentinel
imagery. Export each scene from Google Earth Engine into its pair directory:

```text
rasters/<event_id>/<aoi_id>/
  S1_pre_<date>.tif       # bands: VV, VH
  S1_peak_<date>.tif
  SENTINEL2_pre_<date>.tif   # 8 bands documented by the case
  SENTINEL2_peak_<date>.tif
  AUX_terrain_gsw.tif
```

All Sentinel rasters must use the 10 m reference grid in EPSG:32652. The baseline
resamples the supplied 30 m auxiliary raster internally.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download the public Sentinel-1 weights once during development:

```bash
python scripts/download_sturm_s1_weights.py
```

The archive is approximately 1.8 GB and is verified against the checksum published
by Zenodo before extraction.

The downloaded weights must be packaged with the final solution because evaluation
is performed without internet access.

The public checkpoint was trained for two Sentinel-1 channels. The eight-channel
input is split by date into two four-channel views (`VV, VH, NDWI, MNDWI`) and both
views pass independently through one shared STURM branch. The learned VV and VH
filters are retained; the two optical kernels start at zero. This preserves the
original water prediction separately for `pre` and `peak`. Their features and
feature difference feed a new flood head. This is a safe warm start, not a trained
multimodal model. Fine-tuning on event-level training folds is required before the
optical channels and the new flood head contribute useful predictions.

## Run

```bash
hydrowatch-baseline run \
  --data-root /path/to/hydrowatch_amur \
  --output-dir outputs/sturm_baseline
```

Calculate the official local metric against the supplied reference statistics:

```bash
hydrowatch-baseline score \
  --data-root /path/to/hydrowatch_amur \
  --submission outputs/sturm_baseline/submission.csv
```

The STURM normalization and fusion settings are deliberately explicit in
`configs/sturm_baseline.toml`. They are baseline assumptions and must be selected
using event-level validation, not fitted directly to every open reference mask.

See `THIRD_PARTY_SOURCES.md` for provenance, license and competition-use limits.
