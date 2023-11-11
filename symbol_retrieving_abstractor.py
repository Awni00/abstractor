import tensorflow as tf
from abstracters import RelationalAbstracterLayer
import numpy as np

# TODO: decide on how to integrate this into code-base and name of module
# maybe just integrate into abstracters.py?

class SymbolRetrievingAbstractor(tf.keras.layers.Layer):
    """
    An implementation of the Symbol-Retrieving Abstractor module.

    1) Retrieve symbols
    2) Relational cross-attention
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        dff,
        n_symbols,
        symbol_n_heads=1,
        symbol_binding_dim=None,
        rel_activation_function='softmax',
        use_self_attn=True,
        dropout_rate=0.1,
        **kwargs):
        """
        Parameters
        ----------
        num_layers : int
            number of layers
        num_heads : int
            number of 'heads' in relational cross-attention (relation dimension)
        dff : int
            dimension of intermediate layer in feedforward network
        n_symbols : int
            number of symbols
        symbol_dim : int
            dimension of symbols
        binding_dim : int
            dimension of binding symbols
        rel_activation_function : str, optional
            activation of MHA in relational cross-attention, by default 'softmax'
        use_self_attn : bool, optional
            whether to apply self-attention in addition to relational cross-attn, by default True
        dropout_rate : float, optional
            dropout rate, by default 0.1
        **kwargs : dict
            kwargs for parent Layer class
        """

        super(SymbolRetrievingAbstractor, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.n_symbols = n_symbols
        self.symbol_n_heads = symbol_n_heads
        self.symbol_binding_dim = symbol_binding_dim
        self.rel_activation_function = rel_activation_function
        self.use_self_attn = use_self_attn
        self.dropout_rate = dropout_rate

        # NOTE: we choose symbol_dim to be the same as d_model
        # this is required for residual connection to work
        # TODO think about whether this should be adjusted...

    def build(self, input_shape):

        _, self.sequence_length, self.d_model = input_shape

        self.symbol_retrieval = MultiHeadSymbolRetriever(
            n_heads=self.symbol_n_heads, n_symbols=self.n_symbols,
            symbol_dim=self.d_model, binding_dim=self.symbol_binding_dim)

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.abstracter_layers = [
            RelationalAbstracterLayer(d_model=self.d_model, num_heads=self.num_heads,
                dff=self.dff, mha_activation_type=self.rel_activation_function, use_self_attn=self.use_self_attn,
                dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, inputs):

        symbol_seq = self.symbol_retrieval(inputs) # retrieve symbols

        symbol_seq = self.dropout(symbol_seq)

        for i in range(self.num_layers):
            symbol_seq = self.abstracter_layers[i](symbol_seq, inputs)

        return symbol_seq

class SymbolRetriever(tf.keras.layers.Layer):
    def __init__(self, n_symbols, symbol_dim, binding_dim=None, use_bias=False, softmax_scaler=None, symbol_initializer='random_normal', **kwargs):
        super(SymbolRetriever, self).__init__(**kwargs)

        self.n_symbols = n_symbols
        self.symbol_dim = symbol_dim
        self.binding_dim = binding_dim if binding_dim is not None else symbol_dim
        self.use_bias = use_bias
        self.symbol_initializer = symbol_initializer
        self.softmax_scaler = softmax_scaler if softmax_scaler is not None else 1/np.sqrt(binding_dim)

    def build(self, input_shape):
        self.symbols = self.add_weight(
            name='symbols',
            shape=(self.n_symbols, self.symbol_dim),
            initializer=self.symbol_initializer,
            trainable=True)
        self.binding = self.add_weight(
            name='binding',
            shape=(self.n_symbols, self.binding_dim),
            initializer=self.symbol_initializer,
            trainable=True)
        self.query_mapping = tf.keras.layers.Dense(self.binding_dim, use_bias=self.use_bias, name='query_mapping')

    def call(self, inputs):
        input_keys = self.query_mapping(inputs)
        binding_matrix = tf.einsum('bik,jk->bij', input_keys, self.binding)
        normalized_binding_matrix = tf.nn.softmax(self.softmax_scaler * binding_matrix, axis=-1)
        retrieved_symbols = tf.einsum('bij,jk->bik', normalized_binding_matrix, self.symbols)
        return retrieved_symbols

class MultiHeadSymbolRetriever(tf.keras.layers.Layer):
    def __init__(self, n_heads, n_symbols, symbol_dim, binding_dim=None, symbol_initializer='random_normal', **kwargs):
        super(MultiHeadSymbolRetriever, self).__init__(**kwargs)

        self.n_heads = n_heads
        self.n_symbols = n_symbols
        self.symbol_dim = symbol_dim
        self.binding_dim = binding_dim
        self.symbol_initializer = symbol_initializer
        if symbol_dim % n_heads != 0:
            raise ValueError(f'symbol_dim ({symbol_dim}) must be divisible by num_heads ({n_heads})')

    def build(self, input_shape):
        self.symbol_retrievers = [
            SymbolRetriever(
                n_symbols=self.n_symbols, symbol_dim=self.symbol_dim // self.n_heads,
                binding_dim=self.binding_dim, symbol_initializer=self.symbol_initializer,
                name=f'symbol_retriever_h{i}')
            for i in range(self.n_heads)]

    def call(self, inputs):
        retrieved_symbols = tf.concat([
            self.symbol_retrievers[i](inputs)
            for i in range(self.n_heads)
        ], axis=-1)
        return retrieved_symbols
