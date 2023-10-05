"""Module implementing the multi-attention Decoder: a decoder which iteratively attends over several context sequences"""

import tensorflow as tf
from transformer_modules import DecoderLayer

class MultiAttentionDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, dff, dropout_rate=0.1, name="decoder"):
        """create a MultiAttentionDecoder layer.

        The multi-attention decoder is a variant of the decoder which cross-attends to several context sequences.
        For each layer and for each context sequence, the decoder performs causal self-attention,
        then cross-attention to the context sequence, then processes the result with a feed-forward network.

        Parameters
        ----------
        num_layers : int
            number of 'layers' in the decoder. At each layer, the decoder attends to each context sequence.
        num_heads : int
            number of heads in attention layers.
        dff : int
            intermediate dimension in feedforward networks.
        dropout_rate : float, optional
            dropout rate, by default 0.1
        name : str, optional
            name of layer, by default "decoder"
        """
        super(MultiAttentionDecoder, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def build(self, input_shapes):
        input_shape = input_shapes[0]
        context_shapes = input_shapes[1:]
        self.num_contexts = len(context_shapes)

        _, self.sequence_length, self.d_model = input_shape

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.dec_layers = [
            [DecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate,
            ) for _ in range(self.num_contexts)]
            for _ in range(self.num_layers)
        ]

        self.last_attn_scores = None

    def call(self, inputs):
        x = inputs[0]
        contexts = inputs[1:]

        x = self.dropout(x)

        for i in range(self.num_layers):
            for j, context in enumerate(contexts):
                x = self.dec_layers[i][j](x, context)

        return x
