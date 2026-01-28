"""
Convolutional autoencoder model definition + scoring helpers.
"""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def build_autoencoder(input_shape: Tuple[int, int, int] = (256, 256, 3)) -> tf.keras.Model:
    """
    A straightforward conv autoencoder that works well as a baseline.

    Output uses sigmoid to produce [0,1] images.
    """
    inputs = tf.keras.Input(shape=input_shape, name="image")

    # Encoder
    x = inputs
    for filters in [32, 64, 128, 256]:
        x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(512, 3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)

    # Decoder
    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="reconstruction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="conv_autoencoder")
    return model


@tf.function
def reconstruction_error_map(x: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
    """
    Per-pixel reconstruction error map (MSE averaged across channels).

    Args:
        x: [B,H,W,C]
        x_hat: [B,H,W,C]

    Returns:
        [B,H,W] float32
    """
    return tf.reduce_mean(tf.square(x - x_hat), axis=3)


def _score_from_error_map(
    err_map: tf.Tensor,
    *,
    method: str = "mean",
    topk_percent: float = 1.0,
    topk_min_pixels: int = 100,
) -> tf.Tensor:
    """
    Convert per-pixel error maps into per-image scores.

    Args:
        err_map: [B,H,W] float32
        method: "mean" or "topk"
        topk_percent: percentage of pixels to average (top-k)
        topk_min_pixels: minimum number of pixels to include in top-k
    """
    method = str(method or "mean").lower()
    if method != "topk":
        return tf.reduce_mean(err_map, axis=[1, 2])

    shape = tf.shape(err_map)
    total_pixels = tf.maximum(shape[1] * shape[2], 1)
    pct = tf.constant(float(topk_percent) / 100.0, dtype=tf.float32)
    k_float = tf.cast(total_pixels, tf.float32) * pct
    k = tf.cast(tf.round(k_float), tf.int32)
    k = tf.maximum(k, tf.constant(int(topk_min_pixels), dtype=tf.int32))
    k = tf.minimum(k, total_pixels)
    k = tf.maximum(k, 1)

    flat = tf.reshape(err_map, [shape[0], total_pixels])
    topk = tf.math.top_k(flat, k=k, sorted=False).values
    return tf.reduce_mean(topk, axis=1)


@tf.function
def score_from_reconstruction(
    x: tf.Tensor,
    x_hat: tf.Tensor,
    *,
    method: str = "mean",
    topk_percent: float = 1.0,
    topk_min_pixels: int = 100,
) -> tf.Tensor:
    err_map = reconstruction_error_map(x, x_hat)
    return _score_from_error_map(
        err_map,
        method=method,
        topk_percent=topk_percent,
        topk_min_pixels=topk_min_pixels,
    )


@tf.function
def score_batch(
    model: tf.keras.Model,
    x: tf.Tensor,
    *,
    method: str = "mean",
    topk_percent: float = 1.0,
    topk_min_pixels: int = 100,
) -> tf.Tensor:
    x_hat = model(x, training=False)
    return score_from_reconstruction(
        x,
        x_hat,
        method=method,
        topk_percent=topk_percent,
        topk_min_pixels=topk_min_pixels,
    )
