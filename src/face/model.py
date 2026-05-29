"""CNN architectures: face classification embedding, metric-learning encoder, emotion & liveness."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

WeightsArg = Optional[str]


def _freeze_base_layers(base: keras.Model, train_last_n: int) -> None:
    base.trainable = True
    for layer in base.layers[:-train_last_n]:
        layer.trainable = False


def build_face_classifier(
    img_size: int,
    num_classes: int,
    embedding_dim: int,
    train_last_n: int = 40,
    weights: WeightsArg = "imagenet",
) -> tuple[keras.Model, keras.Model]:
    """
    Classification-based embedding (PDF §2.1): GAP → Dense(embedding) → softmax.
    Returns (classifier, embedding_extractor) sharing weights.
    """
    base = MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
    )
    _freeze_base_layers(base, train_last_n)

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    emb = layers.Dense(embedding_dim, name="embedding")(x)
    logits = layers.Dense(num_classes, activation="softmax", name="softmax")(emb)

    classifier = keras.Model(base.input, logits, name="face_classifier")
    embedding_net = keras.Model(base.input, emb, name="face_classifier_embedding")
    return classifier, embedding_net


def build_metric_encoder(
    img_size: int,
    embedding_dim: int,
    train_last_n: int = 40,
    l2_normalize: bool = True,
    weights: WeightsArg = "imagenet",
) -> keras.Model:
    """
    Metric-learning backbone + embedding; optional L2 normalize for cosine-friendly space.
    """
    base = MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
    )
    _freeze_base_layers(base, train_last_n)

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    emb = layers.Dense(embedding_dim, name="embedding")(x)
    if l2_normalize:
        emb = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2_norm")(emb)
    return keras.Model(base.input, emb, name="face_metric_encoder")


class TripletTrainer(keras.Model):
    """Triplet loss on shared encoder (PDF §2.1 metric learning)."""

    def __init__(self, encoder: keras.Model, margin: float = 0.25):
        super().__init__()
        self.encoder = encoder
        self.margin = margin

    def train_step(self, data):
        (xa, xp, xn), _ = data
        with tf.GradientTape() as tape:
            ea = self.encoder(xa, training=True)
            ep = self.encoder(xp, training=True)
            en = self.encoder(xn, training=True)
            d_ap = tf.reduce_sum(tf.square(ea - ep), axis=-1)
            d_an = tf.reduce_sum(tf.square(ea - en), axis=-1)
            loss = tf.reduce_mean(tf.nn.relu(d_ap - d_an + self.margin))
        vars_ = self.encoder.trainable_variables
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))
        return {"loss": loss}

    def test_step(self, data):
        (xa, xp, xn), _ = data
        ea = self.encoder(xa, training=False)
        ep = self.encoder(xp, training=False)
        en = self.encoder(xn, training=False)
        d_ap = tf.reduce_sum(tf.square(ea - ep), axis=-1)
        d_an = tf.reduce_sum(tf.square(ea - en), axis=-1)
        loss = tf.reduce_mean(tf.nn.relu(d_ap - d_an + self.margin))
        return {"loss": loss}


def build_emotion_classifier(
    img_size: int,
    num_emotions: int,
    train_last_n: int = 50,
    weights: WeightsArg = "imagenet",
) -> keras.Model:
    """Fine-grained emotion head (PDF §4). Train on folder-per-emotion data."""
    base = MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
    )
    _freeze_base_layers(base, train_last_n)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_emotions, activation="softmax", name="emotion")(x)
    return keras.Model(base.input, out, name="emotion_classifier")


def build_liveness_classifier(
    img_size: int,
    train_last_n: int = 50,
    weights: WeightsArg = "imagenet",
) -> keras.Model:
    """Binary live vs spoof (PDF §3). Train on live/ vs spoof/ folders."""
    base = MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
    )
    _freeze_base_layers(base, train_last_n)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="live")(x)
    return keras.Model(base.input, out, name="liveness_classifier")
