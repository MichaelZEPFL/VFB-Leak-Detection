"""
Training pipeline for the convolutional autoencoder.

- Loads images from data/normal/
- train/val split
- training augmentation (mild)
- computes reconstruction-error threshold from val set
- saves artifacts to model/
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .data import list_image_files
from .model import build_autoencoder, score_batch


AUTOTUNE = tf.data.AUTOTUNE


def _load_and_preprocess(path: tf.Tensor, image_w: int, image_h: int) -> tf.Tensor:
    """
    Reads an image from disk and returns float32 [H,W,3] in [0,1].
    """
    bytes_ = tf.io.read_file(path)
    img = tf.io.decode_image(bytes_, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_h, image_w], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _make_augmenter(aug_cfg: Dict[str, Any]) -> tf.keras.Model:
    # Keras preprocessing layers handle rotation/translation reliably.
    rotation = float(aug_cfg.get("rotation", 0.0))
    translation = float(aug_cfg.get("translation", 0.0))
    contrast = float(aug_cfg.get("contrast", 0.0))
    brightness = float(aug_cfg.get("brightness", 0.0))

    layers = []
    if rotation > 0:
        layers.append(tf.keras.layers.RandomRotation(factor=rotation))
    if translation > 0:
        layers.append(tf.keras.layers.RandomTranslation(height_factor=translation, width_factor=translation))
    if contrast > 0:
        layers.append(tf.keras.layers.RandomContrast(factor=contrast))
    # RandomBrightness is not present in some older TF versions; implement via Lambda.
    if brightness > 0:
        def _jitter(x: tf.Tensor) -> tf.Tensor:
            return tf.clip_by_value(tf.image.random_brightness(x, max_delta=brightness), 0.0, 1.0)
        layers.append(tf.keras.layers.Lambda(_jitter))

    if not layers:
        return tf.keras.Sequential(name="augmenter")  # no-op

    return tf.keras.Sequential(layers, name="augmenter")


def build_datasets(
    image_paths: List[Path],
    *,
    image_w: int,
    image_h: int,
    batch_size: int,
    validation_split: float,
    seed: int,
    augment_cfg: Dict[str, Any],
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    if not image_paths:
        raise ValueError("No images found. Capture normal images first into data/normal/.")

    rng = random.Random(seed)
    paths = image_paths[:]
    rng.shuffle(paths)

    n_total = len(paths)

    # Basic input validation / safety checks.
    if n_total < 2:
        raise ValueError(
            f"Not enough images to train a normal-only model (found {n_total}). "
            "Capture more normal images first (recommended: a few hundred)."
        )
    if not (0.0 < float(validation_split) < 1.0):
        raise ValueError(f"validation_split must be between 0 and 1 (exclusive). Got: {validation_split}")

    n_val = max(1, int(n_total * validation_split))
    n_train = n_total - n_val

    # Ensure at least 1 image in each split.
    if n_train < 1:
        n_train = 1
        n_val = n_total - n_train
    if n_val < 1:
        raise ValueError(
            f"Train/val split produced an empty validation set (n_total={n_total}, validation_split={validation_split}). "
            "Capture more images or reduce validation_split."
        )

    train_paths = [str(p) for p in paths[:n_train]]
    val_paths = [str(p) for p in paths[n_train:]]

    train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
    val_ds = tf.data.Dataset.from_tensor_slices(val_paths)

    def _prep(p: tf.Tensor) -> tf.Tensor:
        return _load_and_preprocess(p, image_w, image_h)

    train_ds = train_ds.map(_prep, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_prep, num_parallel_calls=AUTOTUNE)

    aug_enabled = bool(augment_cfg.get("enabled", True))
    augmenter = _make_augmenter(augment_cfg) if aug_enabled else tf.keras.Sequential(name="augmenter")

    if aug_enabled:
        def _aug(x: tf.Tensor) -> tf.Tensor:
            # training=True makes random transforms active
            return augmenter(x, training=True)
        train_ds = train_ds.map(_aug, num_parallel_calls=AUTOTUNE)

    # Autoencoder target = input
    train_ds = train_ds.map(lambda x: (x, x), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x: (x, x), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, n_train, n_val


def compute_errors(
    model: tf.keras.Model,
    image_ds: tf.data.Dataset,
    *,
    scoring_cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Compute per-image reconstruction errors (MSE).
    image_ds must yield (x, x).
    """
    errors: List[float] = []
    for batch_x, _ in image_ds:
        batch_scores = score_batch(
            model,
            batch_x,
            method=str(scoring_cfg.get("method", "mean")),
            topk_percent=float(scoring_cfg.get("topk_percent", 1.0)),
            topk_min_pixels=int(scoring_cfg.get("topk_min_pixels", 100)),
        ).numpy().tolist()
        errors.extend(batch_scores)
    return np.array(errors, dtype=np.float32)


def train_autoencoder(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths_cfg = cfg["paths"]
    train_cfg = cfg["training"]
    img_cfg = cfg["image"]
    threshold_cfg = cfg["threshold"]
    scoring_cfg = dict(cfg.get("scoring", {}) or {})

    # Best-effort reproducibility across runs.
    # (Does not guarantee full determinism on all hardware/backends.)
    try:
        tf.keras.utils.set_random_seed(int(train_cfg.get("seed", 42)))
    except Exception:
        pass

    normal_dir = Path(paths_cfg["data_normal_dir"])
    model_dir = Path(paths_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(normal_dir)

    train_ds, val_ds, n_train, n_val = build_datasets(
        image_paths,
        image_w=int(img_cfg["width"]),
        image_h=int(img_cfg["height"]),
        batch_size=int(train_cfg["batch_size"]),
        validation_split=float(train_cfg["validation_split"]),
        seed=int(train_cfg["seed"]),
        augment_cfg=dict(train_cfg.get("augment", {})),
    )

    model = build_autoencoder(input_shape=(int(img_cfg["height"]), int(img_cfg["width"]), 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(train_cfg["learning_rate"])),
        loss="mse",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(train_cfg["epochs"]),
        callbacks=callbacks,
        verbose=1,
    )

    val_errors = compute_errors(model, val_ds, scoring_cfg=scoring_cfg)
    percentile = float(threshold_cfg["percentile"])
    threshold = float(np.percentile(val_errors, percentile))

    # Save model
    model_path = model_dir / "autoencoder.keras"
    model.save(model_path)

    # Save threshold + stats
    threshold_path = model_dir / "threshold.json"
    stats_path = model_dir / "stats.json"

    threshold_obj = {
        "percentile": percentile,
        "threshold": threshold,
        "scoring_method": str(scoring_cfg.get("method", "mean")),
        "topk_percent": float(scoring_cfg.get("topk_percent", 1.0)),
        "topk_min_pixels": int(scoring_cfg.get("topk_min_pixels", 100)),
    }
    threshold_path.write_text(json.dumps(threshold_obj, indent=2), encoding="utf-8")

    stats_obj = {
        "n_train": n_train,
        "n_val": n_val,
        "val_error_mean": float(val_errors.mean()),
        "val_error_std": float(val_errors.std()),
        "val_error_min": float(val_errors.min()),
        "val_error_max": float(val_errors.max()),
        "final_train_loss": float(history.history["loss"][-1]) if history.history.get("loss") else None,
        "final_val_loss": float(history.history["val_loss"][-1]) if history.history.get("val_loss") else None,
        "scoring_method": str(scoring_cfg.get("method", "mean")),
        "topk_percent": float(scoring_cfg.get("topk_percent", 1.0)),
        "topk_min_pixels": int(scoring_cfg.get("topk_min_pixels", 100)),
        "model_path": str(model_path.resolve()),
        "threshold_path": str(threshold_path.resolve()),
    }
    stats_path.write_text(json.dumps(stats_obj, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path.resolve()),
        "threshold": threshold,
        "percentile": percentile,
        "stats_path": str(stats_path.resolve()),
    }
