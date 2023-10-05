"""This module implements basic building blocks of transformers"""

import numpy as np
import tensorflow as tf
from attention import GlobalSelfAttention, CausalSelfAttention, CrossAttention

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, dff, dropout_rate=0.1, name="encoder"):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        _, self.sequence_length, self.d_model = input_shape

        self.enc_layers = [
            EncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, dff, dropout_rate=0.1, name="decoder"):
        super(Decoder, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        _, self.sequence_length, self.d_model = input_shape

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.dec_layers = [
            DecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x


def create_positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class AddPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_length=1024, name="add_positional_embedding"):
        super().__init__(name=name)
        self.max_length = max_length

    def build(self, input_shape):
        _, self.seq_length, self.vec_dim = input_shape
        self.max_length = max(self.max_length, self.seq_length)
        self.pos_encoding = create_positional_encoding(length=self.max_length, depth=self.vec_dim)

    def call(self, x):
        length = tf.shape(x)[1]

        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.vec_dim, tf.float32))

        # add positional encoding
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class TeacherForcingAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="teacher_forcing_accuracy", ignore_class=None, **kwargs):
        super(TeacherForcingAccuracy, self).__init__(name=name, **kwargs)
        if ignore_class is None or isinstance(ignore_class, int):
            self.ignore_class = ignore_class
        else:
            raise ValueError("`ignore_class` must be None or an integer")

        self.correct_preds = self.add_weight(name="correct_preds", initializer="zeros")
        self.mask_count = self.add_weight(name="mask_count", initializer="zeros")

    def update_state(self, label, pred, sample_weight=None):
        pred = tf.argmax(pred, axis=2)
        label = tf.cast(label, pred.dtype)
        match = label == pred

        if self.ignore_class is None:
            mask = tf.ones_like(label, dtype=tf.bool)
        else:
            mask = label != self.ignore_class

        match = match & mask

        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        self.correct_preds.assign_add(tf.reduce_sum(match))
        self.mask_count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correct_preds / self.mask_count