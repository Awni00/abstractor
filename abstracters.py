"""
Module Implementing different variants of the 'abstracter'.

The abstracter is a module for transformer-based models which aims to encourage
learning abstract relations.

It is characterized by employing learned input-independent 'symbols' in its computation
and using adjusted cross-attention mechanisms (e.g.: relational attention).

Typically, an abstracter module follows an 'encoder'.
For Seq2Seq models, it may be followed by a decoder.
"""

import tensorflow as tf
from transformer_modules import AddPositionalEmbedding, FeedForward
from attention import GlobalSelfAttention, BaseAttention, RelationalAttention, SymbolicAttention, CrossAttention



class RelationalAbstracter(tf.keras.layers.Layer):
    """
    An implementation of the 'Abstractor' module.

    This implementation uses tensorflow's MultiHeadAttention layer
    to implement relational cross-attention.
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        dff,
        use_pos_embedding=True,
        use_learned_symbols=True,
        mha_activation_type='softmax',
        use_self_attn=True,
        dropout_rate=0.1,
        name=None):
        """
        Parameters
        ----------
        num_layers : int
            number of layers
        num_heads : int
            number of 'heads' in relational cross-attention (relation dimension)
        dff : int
            dimension of intermediate layer in feedforward network
        use_pos_embedding : bool, optional
            whether to add positional embeddings to symbols, by default True
        use_learned_symbols : bool, optional
            whether to use learned symbols or nonparametric positional embeddings, by default True
        mha_activation_type : str, optional
            activation of MHA in relational cross-attention, by default 'softmax'
        use_self_attn : bool, optional
            whether to apply self-attention in addition to relational cross-attn, by default True
        dropout_rate : float, optional
            dropout rate, by default 0.1
        name : str, optional
            name of layer, by default None
        """

        super(RelationalAbstracter, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.mha_activation_type = mha_activation_type
        self.use_pos_embedding = use_pos_embedding
        self.use_learned_symbols = use_learned_symbols
        if not self.use_learned_symbols:
            self.use_pos_embedding = True
        self.use_self_attn = use_self_attn
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        _, self.sequence_length, self.d_model = input_shape

        # define the input-independent symbolic input vector sequence at the decoder
        if self.use_learned_symbols:
            normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            self.symbol_sequence = tf.Variable(
                normal_initializer(shape=(self.sequence_length, self.d_model)),
                name='symbols', trainable=True)

        # layer which adds positional embedding (to be used on symbol sequence)
        if self.use_pos_embedding:
            self.add_pos_embedding = AddPositionalEmbedding()

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.abstracter_layers = [
            RelationalAbstracterLayer(d_model=self.d_model, num_heads=self.num_heads,
                dff=self.dff, mha_activation_type=self.mha_activation_type, use_self_attn=self.use_self_attn,
                dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, inputs):
        # symbol sequence is input independent, so use the same one for all computations in the given batch
        # (this broadcasts the symbol_sequence across all inputs in the batch)
        symbol_seq = tf.zeros_like(inputs)
        if self.use_learned_symbols:
            symbol_seq = symbol_seq + self.symbol_sequence
        # add positional embedding
        if self.use_pos_embedding:
            symbol_seq = self.add_pos_embedding(symbol_seq)

        symbol_seq = self.dropout(symbol_seq)

        for i in range(self.num_layers):
            symbol_seq = self.abstracter_layers[i](symbol_seq, inputs)

#             self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return symbol_seq

class RelationalAbstracterLayer(tf.keras.layers.Layer):
  def __init__(self,
    *,
    d_model,
    num_heads,
    dff,
    use_self_attn=True,
    mha_activation_type='softmax',
    dropout_rate=0.1):

    super(RelationalAbstracterLayer, self).__init__()

    self.use_self_attn = use_self_attn

    if self.use_self_attn:
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            activation_type=mha_activation_type,
            dropout=dropout_rate)

    self.relational_crossattention = RelationalAttention(
        num_heads=num_heads,
        key_dim=d_model,
        activation_type=mha_activation_type,
        dropout=dropout_rate)

    self.dff = dff
    if dff is not None:
        self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    if self.use_self_attn:
        x = self.self_attention(x=x)
    x = self.relational_crossattention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.relational_crossattention.last_attn_scores

    if self.dff is not None:
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.

    return x

class SymbolicAbstracter(tf.keras.layers.Layer):
    """
    A variant of an 'Abstractor' module early in development.

    This variant uses a 'symbolic' attention mechanism, in which
    Q <- S, K <- X, V <- X, where X is the input and S are learned symbols.
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        dff,
        use_pos_embedding=True,
        mha_activation_type='softmax',
        dropout_rate=0.1,
        name='symbolic_abstracter'):

        super(SymbolicAbstracter, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.use_pos_embedding = use_pos_embedding
        self.mha_activation_type = mha_activation_type
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        _, self.sequence_length, self.d_model = input_shape

        # define the input-independent symbolic input vector sequence at the decoder
        normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.symbol_sequence = tf.Variable(
            normal_initializer(shape=(self.sequence_length, self.d_model)),
            name='symbols', trainable=True)

        # layer which adds positional embedding (to be used on symbol sequence)
        if self.use_pos_embedding:
            self.add_pos_embedding = AddPositionalEmbedding()

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.abstracter_layers = [
            SymbolicAbstracterLayer(d_model=self.d_model, num_heads=self.num_heads,
                dff=self.dff, mha_activation_type=self.mha_activation_type,
                dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, encoder_context):
        # symbol sequence is input independent, so use the same one for all computations in the given batch
        symbol_seq = tf.zeros_like(encoder_context) + self.symbol_sequence

        # add positional embedding
        if self.use_pos_embedding:
            symbol_seq = self.add_pos_embedding(symbol_seq)


        symbol_seq = self.dropout(symbol_seq)


        for i in range(self.num_layers):
            symbol_seq = self.abstracter_layers[i](symbol_seq, encoder_context)

#             self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return symbol_seq

class SymbolicAbstracterLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        dff,
        mha_activation_type='softmax',
        dropout_rate=0.1,
        name=None):

        super(SymbolicAbstracterLayer, self).__init__(name=name)

        self.mha_activation_type = mha_activation_type

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            activation_type=mha_activation_type,
            dropout=dropout_rate)

        self.symbolic_attention = SymbolicAttention(
            num_heads=num_heads,
            key_dim=d_model,
            activation_type=mha_activation_type,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.self_attention(x=x)
        x = self.symbolic_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.symbolic_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.

        return x


class AblationAbstractor(tf.keras.layers.Layer):
    """
    An 'Ablation' Abstractor model.

    This model is the same as the RelationalAbstractor, but uses
    standard cross-attention instead of relational cross-attention.
    This is used to isolate for the effect of the cross-attention scheme
    in experiments.
    """
    def __init__(
        self,
        num_layers,
        num_heads,
        dff,
        use_self_attn=True,
        use_pos_embedding=True,
        mha_activation_type='softmax',
        dropout_rate=0.1,
        name='ablation_model'):

        super(AblationAbstractor, self).__init__(name=name)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.use_self_attn = use_self_attn
        self.mha_activation_type = mha_activation_type
        self.use_pos_embedding = use_pos_embedding
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        _, self.sequence_length, self.d_model = input_shape

        # define the input-independent symbolic input vector sequence at the decoder
        normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.symbol_sequence = tf.Variable(
            normal_initializer(shape=(self.sequence_length, self.d_model)),
            name='symbols', trainable=True)

        # layer which adds positional embedding (to be used on symbol sequence)
        if self.use_pos_embedding:
            self.add_pos_embedding = AddPositionalEmbedding()

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.abstracter_layers = [
            AblationAbstractorLayer(d_model=self.d_model, num_heads=self.num_heads,
                dff=self.dff, use_self_attn=self.use_self_attn,
                mha_activation_type=self.mha_activation_type,
                dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, encoder_context):
        # symbol sequence is input independent, so use the same one for all computations in the given batch
        symbol_seq = tf.zeros_like(encoder_context) + self.symbol_sequence

        # add positional embedding
        if self.use_pos_embedding:
            symbol_seq = self.add_pos_embedding(symbol_seq)

        symbol_seq = self.dropout(symbol_seq)

        for i in range(self.num_layers):
            symbol_seq = self.abstracter_layers[i](symbol_seq, encoder_context)

#             self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return symbol_seq

class AblationAbstractorLayer(tf.keras.layers.Layer):
  def __init__(self,
    *,
    d_model,
    num_heads,
    dff,
    use_self_attn=True,
    mha_activation_type='softmax',
    dropout_rate=0.1):

    super(AblationAbstractorLayer, self).__init__()

    self.use_self_attn = use_self_attn

    if use_self_attn:
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

    self.crossattention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        activation_type=mha_activation_type,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    if self.use_self_attn:
        x = self.self_attention(x=x)
    x = self.crossattention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.crossattention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x