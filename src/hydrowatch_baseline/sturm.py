from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np


INPUT_CHANNELS = (
    "vv_pre", "vh_pre", "vv_peak", "vh_peak",
    "ndwi_pre", "mndwi_pre", "ndwi_peak", "mndwi_peak",
)
OUTPUT_CHANNELS = ("water_pre", "water_peak", "flood")


def _scale_db(values: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _scale_index(values: np.ndarray, missing_value: float) -> np.ndarray:
    valid = np.isfinite(values)
    result = np.full(values.shape, missing_value, dtype=np.float32)
    result[valid] = np.clip((values[valid] + 1.0) / 2.0, 0.0, 1.0)
    return result


def build_eight_channel_input(
    vv_pre: np.ndarray,
    vh_pre: np.ndarray,
    vv_peak: np.ndarray,
    vh_peak: np.ndarray,
    ndwi_pre: np.ndarray,
    mndwi_pre: np.ndarray,
    ndwi_peak: np.ndarray,
    mndwi_peak: np.ndarray,
    config: dict,
) -> np.ndarray:
    arrays = (vv_pre, vh_pre, vv_peak, vh_peak, ndwi_pre, mndwi_pre, ndwi_peak, mndwi_peak)
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(f"All eight channels must have one grid, got {[x.shape for x in arrays]}")
    limits = config["normalization"]
    missing = config["model"]["missing_optical_value"]
    channels = (
        _scale_db(vv_pre, limits["vv_min_db"], limits["vv_max_db"]),
        _scale_db(vh_pre, limits["vh_min_db"], limits["vh_max_db"]),
        _scale_db(vv_peak, limits["vv_min_db"], limits["vv_max_db"]),
        _scale_db(vh_peak, limits["vh_min_db"], limits["vh_max_db"]),
        _scale_index(ndwi_pre, missing),
        _scale_index(mndwi_pre, missing),
        _scale_index(ndwi_peak, missing),
        _scale_index(mndwi_peak, missing),
    )
    return np.stack(channels, axis=-1).astype(np.float32)


def _import_sturm_model(repository: Path):
    arch_dir = repository / "arch"
    if not (arch_dir / "model.py").exists():
        raise FileNotFoundError(f"STURM model code not found at {arch_dir}")
    sys.path.insert(0, str(arch_dir))
    try:
        from model import unet_model
    finally:
        sys.path.pop(0)
    return unet_model


def _transfer_sturm_weights(source, feature_backbone, water_head, flood_head) -> None:
    """Warm-start the shared four-channel branch from the two-channel model."""
    source_layers = [layer for layer in source.layers if layer.get_weights()]
    target_body = [layer for layer in feature_backbone.layers if layer.get_weights()]
    source_head = source_layers[-1]
    source_body = source_layers[:-1]
    if len(source_body) != len(target_body):
        raise ValueError("STURM source and multimodal target bodies are structurally different")

    expanded_first_conv = False
    for source_layer, target_layer in zip(source_body, target_body, strict=True):
        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        if all(a.shape == b.shape for a, b in zip(source_weights, target_weights, strict=True)):
            target_layer.set_weights(source_weights)
            continue
        source_kernel, target_kernel = source_weights[0], target_weights[0]
        compatible_first_conv = (
            not expanded_first_conv
            and source_kernel.ndim == 4
            and source_kernel.shape[2] == 2
            and target_kernel.shape[2] == 4
            and source_kernel.shape[:2] == target_kernel.shape[:2]
            and source_kernel.shape[3] == target_kernel.shape[3]
        )
        if not compatible_first_conv:
            raise ValueError(
                f"Cannot transfer {source_layer.name} {source_kernel.shape} to "
                f"{target_layer.name} {target_kernel.shape}"
            )
        target_kernel.fill(0.0)
        target_kernel[:, :, 0, :] = source_kernel[:, :, 0, :]
        target_kernel[:, :, 1, :] = source_kernel[:, :, 1, :]
        target_layer.set_weights([target_kernel, source_weights[1]])
        expanded_first_conv = True

    source_kernel, source_bias = source_head.get_weights()
    target_kernel, target_bias = water_head.get_weights()
    water_kernel = source_kernel[..., 1] - source_kernel[..., 0]
    water_bias = source_bias[1] - source_bias[0]
    target_kernel[..., 0] = water_kernel
    target_bias[0] = water_bias
    water_head.set_weights([target_kernel, target_bias])

    flood_kernel, flood_bias = flood_head.get_weights()
    flood_kernel.fill(0.0)
    flood_bias[0] = -4.0
    flood_head.set_weights([flood_kernel, flood_bias])


def load_multimodal_model(repository: Path, weights: Path, patch_size: int):
    if not weights.exists():
        raise FileNotFoundError(
            f"STURM weights not found at {weights}. Run scripts/download_sturm_s1_weights.py first."
        )
    unet_model = _import_sturm_model(repository)
    source = unet_model(
        n_classes=2, tile_width=patch_size, tile_height=patch_size, n_bands=2,
        n_blocks=6, class_weight_list=[1, 1], normalize_inputs=False,
    )
    source.load_weights(weights)
    import tensorflow as tf

    # Each date is processed independently by the same four-channel branch:
    # VV, VH, NDWI, MNDWI. Sharing the branch preserves the S1 checkpoint's
    # behavior for both dates and keeps the parameter count manageable.
    branch_base = unet_model(
        n_classes=2, tile_width=patch_size, tile_height=patch_size, n_bands=4,
        n_blocks=6, class_weight_list=[1, 1], normalize_inputs=False,
    )
    feature_backbone = tf.keras.Model(
        branch_base.input, branch_base.layers[-2].output, name="shared_sturm_backbone"
    )
    inputs = tf.keras.Input((patch_size, patch_size, len(INPUT_CHANNELS)), name="multimodal_input")
    pre = tf.keras.layers.Lambda(
        lambda values: tf.gather(values, [0, 1, 4, 5], axis=-1), name="pre_channels"
    )(inputs)
    peak = tf.keras.layers.Lambda(
        lambda values: tf.gather(values, [2, 3, 6, 7], axis=-1), name="peak_channels"
    )(inputs)
    pre_features = feature_backbone(pre)
    peak_features = feature_backbone(peak)

    water_head = tf.keras.layers.Conv2D(1, 1, padding="same", name="shared_water_logit")
    water_pre = tf.keras.layers.Activation("sigmoid", name="water_pre")(water_head(pre_features))
    water_peak = tf.keras.layers.Activation("sigmoid", name="water_peak")(water_head(peak_features))
    change = tf.keras.layers.Subtract(name="feature_change")([peak_features, pre_features])
    flood_features = tf.keras.layers.Concatenate(name="flood_features")(
        [pre_features, peak_features, change]
    )
    flood_head = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid", padding="same", name="flood"
    )
    flood = flood_head(flood_features)
    outputs = tf.keras.layers.Concatenate(name="hydro_masks")([water_pre, water_peak, flood])
    target = tf.keras.Model(inputs, outputs, name="sturm_multimodal_8ch")
    _transfer_sturm_weights(source, feature_backbone, water_head, flood_head)
    return target


def predict_multimask(
    image: np.ndarray,
    model,
    patch_size: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != len(INPUT_CHANNELS):
        raise ValueError(f"Expected HxWx8 input, got {image.shape}")
    if not 0 < stride <= patch_size:
        raise ValueError("stride must be in the interval (0, patch_size]")
    height, width, _ = image.shape
    padded_h = max(patch_size, math.ceil(max(0, height - patch_size) / stride) * stride + patch_size)
    padded_w = max(patch_size, math.ceil(max(0, width - patch_size) / stride) * stride + patch_size)
    padded = np.pad(image, ((0, padded_h - height), (0, padded_w - width), (0, 0)), mode="reflect")
    ys = list(range(0, padded_h - patch_size + 1, stride))
    xs = list(range(0, padded_w - patch_size + 1, stride))
    total = np.zeros((padded_h, padded_w, len(OUTPUT_CHANNELS)), dtype=np.float32)
    count = np.zeros((padded_h, padded_w, 1), dtype=np.float32)
    coordinates = [(y, x) for y in ys for x in xs]
    for start in range(0, len(coordinates), batch_size):
        batch_coords = coordinates[start : start + batch_size]
        batch = np.stack([padded[y : y + patch_size, x : x + patch_size] for y, x in batch_coords])
        probabilities = model.predict(batch, batch_size=batch_size, verbose=0)
        for (y, x), prediction in zip(batch_coords, probabilities, strict=True):
            total[y : y + patch_size, x : x + patch_size] += prediction
            count[y : y + patch_size, x : x + patch_size] += 1.0
    return (total / np.maximum(count, 1.0))[:height, :width]
