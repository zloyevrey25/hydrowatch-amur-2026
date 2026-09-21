# HydroWatch Amur baseline

This project adapts the public STURM-Flood Sentinel-1 U-Net as the first baseline
for the CosmoHackathon 2026 hydrological monitoring case.

The baseline performs the following steps:

1. reads VV and VH sigma0 bands for the `pre` and `peak` dates;
2. normalizes them to the STURM input range;
3. runs overlapping 128 x 128 STURM U-Net inference;
4. optionally fuses the SAR probability with the supplied Sentinel-2 indices;
5. removes implausible pixels using slope, HAND, permanent water and built-up layers;
6. derives new flooding as `water_peak & ~water_pre & ~permanent`;
7. writes one flood GeoTIFF per pair and `submission.csv`;
8. computes the local competition Score and all four components.

## Expected data

Extract the case archive so that `pairs.csv`, `reference_masks/`, `vectors/` and
`rasters/` are under one data root. The lite archive does not contain Sentinel
imagery. Export each scene from Google Earth Engine into its pair directory:

```text
rasters/<event_id>/<aoi_id>/
  S1_pre_<date>.tif       # bands: VV, VH (optional third ratio band is ignored)
  S1_peak_<date>.tif
  SENTINEL2_pre_<date>.tif   # optional, 8 bands documented by the case
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
