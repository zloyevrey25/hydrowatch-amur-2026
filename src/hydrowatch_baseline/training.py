from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import tensorflow as tf

from .dataset import select_training_rows
from .pipeline import find_single
from .sturm import build_eight_channel_input, load_multimodal_model


def assert_imagery_ready(data_root: Path, rows: pd.DataFrame) -> None:
    missing: list[str] = []
    pairs = pd.read_csv(data_root / "pairs.csv").set_index("pair_id")
    for pair_id in sorted(rows["pair_id"].unique()):
        directory = data_root / pairs.loc[pair_id, "rasters_dir"]
        if find_single(directory, "S1_pre_*.tif", required=False) is None:
            missing.append(f"{pair_id}: S1_pre")
        if find_single(directory, "S1_peak_*.tif", required=False) is None:
            missing.append(f"{pair_id}: S1_peak")
    if missing:
        preview = "\n".join(missing[:12])
        suffix = "\n..." if len(missing) > 12 else ""
        raise FileNotFoundError(
            "Sentinel-1 GeoTIFFs are required before training. Missing:\n"
            f"{preview}{suffix}"
        )


class PatchSequence(tf.keras.utils.Sequence):
    """Disk-backed Keras sequence for aligned eight-channel raster patches."""

    def __init__(
        self,
        data_root: Path,
        rows: pd.DataFrame,
        config: dict,
        batch_size: int,
        shuffle: bool,
        optical_dropout: float,
        augment: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.rows = rows.reset_index(drop=True)
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.optical_dropout = optical_dropout
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.rows))
        self.pairs = pd.read_csv(data_root / "pairs.csv").set_index("pair_id")
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)

    def _read_patch(self, record) -> tuple[np.ndarray, np.ndarray]:
        import rasterio
        from rasterio.windows import Window

        pair = self.pairs.loc[record.pair_id]
        directory = self.data_root / pair.rasters_dir
        size = int(record.patch_size)
        window = Window(int(record.col), int(record.row), size, size)

        def read_s1(path: Path):
            with rasterio.open(path) as source:
                return source.read(
                    [1, 2], window=window, out_dtype="float32", masked=True
                ).filled(np.nan)

        s1_pre = read_s1(find_single(directory, "S1_pre_*.tif"))
        s1_peak = read_s1(find_single(directory, "S1_peak_*.tif"))

        def read_s2(pattern: str):
            path = find_single(directory, pattern, required=False)
            if path is None or self.rng.random() < self.optical_dropout:
                return np.full((2, size, size), np.nan, dtype=np.float32)
            bands = [self.config["optical"]["ndwi_band"], self.config["optical"]["mndwi_band"]]
            with rasterio.open(path) as source:
                return source.read(
                    bands, window=window, out_dtype="float32", masked=True
                ).filled(np.nan)

        s2_pre = read_s2("SENTINEL2_pre_*.tif")
        s2_peak = read_s2("SENTINEL2_peak_*.tif")
        image = build_eight_channel_input(
            s1_pre[0], s1_pre[1], s1_peak[0], s1_peak[1],
            s2_pre[0], s2_pre[1], s2_peak[0], s2_peak[1], self.config,
        )
        reference = self.data_root / pair.reference_mask
        with rasterio.open(reference) as source:
            # Reference order: flood, water_pre, water_peak, permanent, receded.
            target = np.moveaxis(
                source.read([2, 3, 1], window=window, out_dtype="float32"), 0, -1
            )
        if self.augment:
            if self.rng.random() < 0.5:
                image, target = image[:, ::-1], target[:, ::-1]
            if self.rng.random() < 0.5:
                image, target = image[::-1], target[::-1]
        return image, target

    def __getitem__(self, batch_index: int):
        selected = self.order[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
        batch = [self._read_patch(self.rows.iloc[index]) for index in selected]
        images, targets = zip(*batch, strict=True)
        return np.stack(images), np.stack(targets)


def multimask_loss(channel_weights):
    channel_weights = tf.constant(channel_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        bce = tf.reduce_mean(bce, axis=[1, 2])
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        return tf.reduce_mean((bce + dice_loss) * channel_weights)

    loss.__name__ = "weighted_bce_dice"
    return loss


def channel_iou(channel: int, name: str):
    def metric(y_true, y_pred):
        truth = y_true[..., channel] >= 0.5
        prediction = y_pred[..., channel] >= 0.5
        intersection = tf.reduce_sum(tf.cast(truth & prediction, tf.float32))
        union = tf.reduce_sum(tf.cast(truth | prediction, tf.float32))
        return (intersection + 1.0) / (union + 1.0)

    metric.__name__ = name
    return metric


def channel_f1(channel: int, name: str):
    def metric(y_true, y_pred):
        truth = tf.cast(y_true[..., channel] >= 0.5, tf.float32)
        prediction = tf.cast(y_pred[..., channel] >= 0.5, tf.float32)
        true_positive = tf.reduce_sum(truth * prediction)
        denominator = tf.reduce_sum(truth) + tf.reduce_sum(prediction)
        return (2.0 * true_positive + 1.0) / (denominator + 1.0)

    metric.__name__ = name
    return metric


def compile_model(model, learning_rate: float, channel_weights) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=multimask_loss(channel_weights),
        metrics=[
            channel_iou(0, "iou_water_pre"),
            channel_iou(1, "iou_water_peak"),
            channel_iou(2, "iou_flood"),
            channel_f1(0, "f1_water_pre"),
            channel_f1(1, "f1_water_peak"),
            channel_f1(2, "f1_flood"),
        ],
    )


def set_warmup_trainable(model) -> None:
    backbone = model.get_layer("shared_sturm_backbone")
    for layer in backbone.layers:
        layer.trainable = False
    first_conv = next(
        layer for layer in backbone.layers
        if layer.get_weights() and layer.get_weights()[0].ndim == 4
    )
    first_conv.trainable = True


def train_multimodal(
    data_root: Path,
    manifest_path: Path,
    output_dir: Path,
    repository: Path,
    source_weights: Path,
    config: dict,
    validation_event: str,
    warmup_epochs: int = 3,
    finetune_epochs: int = 12,
    batch_size: int = 2,
    seed: int = 42,
    max_train_patches: int | None = None,
    max_validation_patches: int | None = None,
) -> Path:
    tf.keras.utils.set_random_seed(seed)
    training_config = config["training"]
    if training_config.get("mixed_precision", False) and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    manifest = pd.read_csv(manifest_path)
    train_rows, validation_rows = select_training_rows(
        manifest, validation_event, seed, training_config["negatives_per_positive"]
    )
    if max_train_patches is not None:
        train_rows = train_rows.head(max_train_patches).copy()
    if max_validation_patches is not None:
        validation_rows = validation_rows.head(max_validation_patches).copy()
    if train_rows.empty or validation_rows.empty:
        raise ValueError("Training and validation patch selections must both be non-empty")
    assert_imagery_ready(data_root, pd.concat([train_rows, validation_rows]))
    train_sequence = PatchSequence(
        data_root, train_rows, config, batch_size, True,
        training_config["optical_dropout"], True, seed,
    )
    validation_sequence = PatchSequence(
        data_root, validation_rows, config, batch_size, False, 0.0, False, seed,
    )
    model = load_multimodal_model(repository, source_weights, config["sturm"]["patch_size"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps({
            "validation_event": validation_event,
            "seed": seed,
            "batch_size": batch_size,
            "train_patches": len(train_rows),
            "validation_patches": len(validation_rows),
            "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history_path = output_dir / "training_history.csv"
    history_path.unlink(missing_ok=True)
    best_weights = output_dir / "best.weights.h5"
    best_weights.unlink(missing_ok=True)
    callbacks = [
        tf.keras.callbacks.CSVLogger(history_path, append=True),
        tf.keras.callbacks.ModelCheckpoint(
            best_weights, monitor="val_loss", save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True,
        ),
    ]
    channel_weights = [
        training_config["water_pre_loss_weight"],
        training_config["water_peak_loss_weight"],
        training_config["flood_loss_weight"],
    ]

    if warmup_epochs > 0:
        set_warmup_trainable(model)
        compile_model(model, training_config["warmup_learning_rate"], channel_weights)
        model.fit(
            train_sequence, validation_data=validation_sequence,
            epochs=warmup_epochs, callbacks=callbacks,
        )

    model.get_layer("shared_sturm_backbone").trainable = True
    compile_model(model, training_config["finetune_learning_rate"], channel_weights)
    model.fit(
        train_sequence, validation_data=validation_sequence,
        initial_epoch=warmup_epochs,
        epochs=warmup_epochs + finetune_epochs,
        callbacks=callbacks,
    )
    if best_weights.exists():
        model.load_weights(best_weights)
    final_weights = output_dir / "final.weights.h5"
    model.save_weights(final_weights)
    return final_weights
