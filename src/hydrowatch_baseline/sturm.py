from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np


def normalize_s1(vv: np.ndarray, vh: np.ndarray, config: dict) -> np.ndarray:
    limits = config["normalization"]

    def scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
        values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
        return np.clip((values - low) / (high - low), 0.0, 1.0)

    vv_scaled = scale(vv, limits["vv_min_db"], limits["vv_max_db"])
    vh_scaled = scale(vh, limits["vh_min_db"], limits["vh_max_db"])
    return np.stack([vv_scaled, vh_scaled], axis=-1).astype(np.float32)


def load_s1_model(repository: Path, weights: Path, patch_size: int):
    arch_dir = repository / "arch"
    if not (arch_dir / "model.py").exists():
        raise FileNotFoundError(f"STURM model code not found at {arch_dir}")
    if not weights.exists():
        raise FileNotFoundError(
            f"STURM weights not found at {weights}. Run scripts/download_sturm_s1_weights.py first."
        )
    sys.path.insert(0, str(arch_dir))
    try:
        from model import unet_model
    finally:
        sys.path.pop(0)

    model = unet_model(
        n_classes=2,
        tile_width=patch_size,
        tile_height=patch_size,
        n_bands=2,
        n_blocks=6,
        class_weight_list=[1, 1],
        normalize_inputs=False,
    )
    model.load_weights(weights)
    return model


def predict_probability(
    image: np.ndarray,
    model,
    patch_size: int,
    stride: int,
    batch_size: int,
    water_class_index: int = 1,
) -> np.ndarray:
    """Run overlapping tiled inference and average probabilities at overlaps."""
    if image.ndim != 3 or image.shape[2] != 2:
        raise ValueError(f"Expected HxWx2 input, got {image.shape}")
    if not 0 < stride <= patch_size:
        raise ValueError("stride must be in the interval (0, patch_size]")

    height, width, _ = image.shape
    padded_h = max(patch_size, math.ceil((height - patch_size) / stride) * stride + patch_size)
    padded_w = max(patch_size, math.ceil((width - patch_size) / stride) * stride + patch_size)
    pad_h, pad_w = padded_h - height, padded_w - width
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    ys = list(range(0, padded_h - patch_size + 1, stride))
    xs = list(range(0, padded_w - patch_size + 1, stride))
    total = np.zeros((padded_h, padded_w), dtype=np.float32)
    count = np.zeros((padded_h, padded_w), dtype=np.float32)

    coordinates = [(y, x) for y in ys for x in xs]
    for start in range(0, len(coordinates), batch_size):
        batch_coords = coordinates[start : start + batch_size]
        batch = np.stack([
            padded[y : y + patch_size, x : x + patch_size] for y, x in batch_coords
        ])
        probabilities = model.predict(batch, batch_size=batch_size, verbose=0)
        for (y, x), prediction in zip(batch_coords, probabilities, strict=True):
            total[y : y + patch_size, x : x + patch_size] += prediction[..., water_class_index]
            count[y : y + patch_size, x : x + patch_size] += 1.0

    return (total / np.maximum(count, 1.0))[:height, :width]

