"""Implements attention mechanisms for Transformers and Abstractors"""

import tensorflow as tf

from multi_head_attention import MultiHeadAttention
# from tensorflow.keras.layers import MultiHeadAttention

class BaseAttention(tf.keras.layers.Layer):
    '''base attention class with support for layer norm and residual connections.'''
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
    '''global self-attention (i.e., non-causal). Q,K,V <- X'''
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
    '''causal self-attention. Q,K,V <- X'''
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
    '''cross-attention. Q <- X, K,V <- context'''

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


class RelationalAttention(BaseAttention):
  '''relational (cross-)attention. Q,K <- X, V <- S'''
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

# the attention mechanism below was an early experimental idea
# this variant used Q <- S, K<-X, V<-S, where S are input-independent symbols.
# essentially, cross-attention to a set of input-independent symbols.
# this did not make it into the paper. 
# should not be confused with "symbol retrieval", although implementation is similar.
# this is only here for "historical" reference
class SymbolicAttention(BaseAttention):
    '''symbolic (cross-)attention. Q <- X, K,V <- S'''
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