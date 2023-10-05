"""Implements cross-attention mechanisms for transformers and Abstractors"""

import tensorflow as tf

from multi_head_attention import MultiHeadAttention
# from tensorflow.keras.layers import MultiHeadAttention

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self,
        use_residual=True,
        use_layer_norm=True,
        **kwargs):

        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        if use_layer_norm: self.layernorm = tf.keras.layers.LayerNormalization()
        if use_residual: self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)

        if self.use_residual:
            x = self.add([x, attn_output])
        else:
            x = attn_output

        if self.use_layer_norm:
            x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)

        if self.use_residual:
            x = self.add([x, attn_output])
        else:
            x = attn_output

        if self.use_layer_norm:
            x = self.layernorm(x)


        return x


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        if self.use_residual:
            x = self.add([x, attn_output])
        else:
            x = attn_output

        if self.use_layer_norm:
            x = self.layernorm(x)


        return x

class SymbolicAttention(BaseAttention):
    def call(self, symbols, inputs):
        attn_output, attn_scores = self.mha(
            query=symbols,
            key=inputs,
            value=symbols ,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        if self.use_residual:
            symbols = self.add([symbols, attn_output])
        else:
            symbols = attn_output

        if self.use_layer_norm:
            symbols = self.layernorm(symbols)


        return symbols

class RelationalAttention(BaseAttention):
  def call(self, symbols, inputs):
    attn_output, attn_scores = self.mha(
        query=inputs,
        key=inputs,
        value=symbols ,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    if self.use_residual:
        symbols = self.add([symbols, attn_output])
    else:
        symbols = attn_output

    if self.use_layer_norm:
        symbols = self.layernorm(symbols)

    return symbols
