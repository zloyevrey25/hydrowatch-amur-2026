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
resamples the supplied 30 m auxiliary raster internally. Sentinel-2 is optional for
individual pairs: absent or cloud-masked optical pixels receive the explicit value
`-1`, and optical dropout during training teaches the network to fall back to SAR.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Container

Build a portable CLI image without case data, model weights or generated outputs:

```bash
docker build -t hydrowatch-amur .
docker run --rm hydrowatch-amur --help
```

Mount local assets when running a command. The paths in the container must match
the repository configuration, so the project directory is mounted at `/workspace`:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  hydrowatch-amur prepare \
  --data-root data/hydrowatch_amur \
  --output-dir outputs/preparation
```

This keeps competition data, downloaded STURM weights and outputs outside the
image. `run` and training additionally require the mounted
`external/STURM-Flood` submodule and `models/sturm_s1` weights. To start
training from the image, replace the CLI entry point:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  --entrypoint python \
  hydrowatch-amur scripts/train.py \
  --data-root data/hydrowatch_amur \
  --validation-event flood_2021_06_amur
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

## Prepare training and validation data

The lite case archive contains 11 reference masks, auxiliary rasters, vectors and
scene passports, but no Sentinel imagery. Export the required Sentinel GeoTIFFs
first. Generate Earth Engine tasks aligned exactly to every reference grid:

```bash
python scripts/generate_gee_exports.py \
  --data-root data/hydrowatch_amur \
  --output outputs/gee_export.js
```

Paste `outputs/gee_export.js` into the Google Earth Engine Code Editor, run it, and
start the generated Drive export tasks. After downloading the GeoTIFFs into one
directory, validate their grid and install them into the expected pair directories:

```bash
python scripts/install_gee_exports.py \
  --source-dir /path/to/downloaded/geotiffs \
  --data-root data/hydrowatch_amur
```

The export uses `COPERNICUS/S1_GRD` in dB with the required orbit and
`COPERNICUS/S2_SR_HARMONIZED` with SCL cloud masking. It writes VV, VH and their dB
difference for Sentinel-1, and B3, B4, B8, B11, NDWI, MNDWI, NDVI and AWEIsh for
Sentinel-2. Pairs with no optical dates remain valid SAR-only training examples.

Then audit the dataset and build patch manifests:

```bash
hydrowatch-baseline prepare \
  --data-root data/hydrowatch_amur \
  --output-dir outputs/preparation
```

This creates:

- `data_audit.csv` with missing files, raster properties and reference areas;
- `event_folds.csv` with one fold per complete hydrological event;
- `patch_manifest.csv` with label fractions for each 128 x 128 patch.
- `dataset_summary.json` with dataset readiness and class-balance statistics.

Splitting by event prevents patches from the same flood and AOI leaking into both
training and validation. Empty patches are downsampled during training, while all
validation patches are retained.

## Train

The required reproducible training entry point is `scripts/train.py`. For example,
hold out the entire June 2021 event:

```bash
python scripts/train.py \
  --data-root data/hydrowatch_amur \
  --manifest outputs/preparation/patch_manifest.csv \
  --validation-event flood_2021_06_amur \
  --output-dir outputs/training/fold_2
```

Training first updates the optical part of the first convolution and the new flood
head while the rest of STURM is frozen. It then fine-tunes the full shared branch at
a lower learning rate. The loss combines binary cross-entropy and Dice, gives extra
weight to the rare flood channel, balances empty patches, and randomly removes the
optical inputs in 35% of training patches. All these values are explicit in
`configs/sturm_baseline.toml`.

## Event-level experiments

Run every leave-one-event-out fold for three comparable feature sets:

```bash
python scripts/run_experiments.py \
  --data-root data/hydrowatch_amur \
  --manifest outputs/preparation/patch_manifest.csv \
  --output-root outputs/experiments
```

The default ablations are `sar-only`, `sar-ndwi`, and `sar-ndwi-mndwi`. Each
run writes weights, predictions, `submission.csv`, a per-pair report, and an
incrementally updated `experiment_summary.csv` / `experiment_summary.json`.
Use `--dry-run` to inspect the fold-by-ablation plan without training; use
`--resume` to reuse an existing `final.weights.h5`.

## Run

```bash
hydrowatch-baseline run \
  --data-root /path/to/hydrowatch_amur \
  --output-dir outputs/sturm_baseline \
  --trained-weights outputs/training/fold_2/best.weights.h5
```

Calculate the official local metric against the supplied reference statistics:

```bash
hydrowatch-baseline score \
  --data-root /path/to/hydrowatch_amur \
  --submission outputs/sturm_baseline/submission.csv
```

Write a reusable validation report with the same score components plus per-pair
area errors and control-pair penalties:

```bash
hydrowatch-baseline report \
  --data-root /path/to/hydrowatch_amur \
  --submission outputs/sturm_baseline/submission.csv \
  --output-dir outputs/report/fold_2
```

This creates `score_summary.json` and `pair_diagnostics.csv`.

The STURM normalization and fusion settings are deliberately explicit in
`configs/sturm_baseline.toml`. They are baseline assumptions and must be selected
using event-level validation, not fitted directly to every open reference mask.

See `THIRD_PARTY_SOURCES.md` for provenance, license and competition-use limits.
