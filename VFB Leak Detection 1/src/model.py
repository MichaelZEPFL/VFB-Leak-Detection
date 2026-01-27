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
def reconstruction_error_mse(x: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
    """
    Per-image reconstruction error (MSE).

    Args:
        x: [B,H,W,C]
        x_hat: [B,H,W,C]

    Returns:
        [B] float32
    """
    return tf.reduce_mean(tf.square(x - x_hat), axis=[1, 2, 3])


@tf.function
def score_batch(model: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
    x_hat = model(x, training=False)
    return reconstruction_error_mse(x, x_hat)
