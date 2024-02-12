"""
Implements an Abstractor module with symbols assigned to each object via symbolic attention.

Symbolic attention retrieves a symbol from a library of learned symbols via attention.
The retrieved symbols are then used in relational cross-attention (instead of positional symbols).
"""

import tensorflow as tf
from abstracters import RelationalAbstracterLayer
from transformer_modules import AddPositionalEmbedding
import numpy as np

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
        add_pos_embedding=True,
        rel_activation_function='softmax',
        use_self_attn=True,
        symbol_retriever_type=1, # NOTE / TODO TEMPORARY
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
        symbol_n_heads : int, optional
            number of heads in SymbolRetriever, by default 1
        symbol_binding_dim : int, optional
            dimension of binding symbols, by default None
        add_pos_embedding : bool, optional
            whether to add positional embeddings to symbols after retrieval, by default True
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
        self.should_add_pos_embedding = add_pos_embedding
        self.rel_activation_function = rel_activation_function
        self.use_self_attn = use_self_attn
        self.symbol_retriever_type = symbol_retriever_type
        self.dropout_rate = dropout_rate

        # NOTE: we choose symbol_dim to be the same as d_model
        # this is required for residual connection to work
        # TODO think about whether this should be adjusted...

    def build(self, input_shape):


        _, self.sequence_length, self.d_model = input_shape

        if self.symbol_retriever_type == 1:
            self.symbol_retrieval = MultiHeadSymbolRetriever(
                n_heads=self.symbol_n_heads, n_symbols=self.n_symbols,
                symbol_dim=self.d_model, binding_dim=self.symbol_binding_dim)
        elif self.symbol_retriever_type == 2:
            self.symbol_retrieval = MultiHeadSymbolRetrieval2(
                n_heads=self.symbol_n_heads, n_symbols=self.n_symbols,
                symbol_dim=self.d_model, binding_dim=self.symbol_binding_dim)

        if self.should_add_pos_embedding: self.add_pos_embedding = AddPositionalEmbedding()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.abstracter_layers = [
            RelationalAbstracterLayer(d_model=self.d_model, num_heads=self.num_heads,
                dff=self.dff, mha_activation_type=self.rel_activation_function, use_self_attn=self.use_self_attn,
                dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, inputs):

        symbol_seq = self.symbol_retrieval(inputs) # retrieve symbols
        if self.should_add_pos_embedding:
            symbol_seq = self.add_pos_embedding(symbol_seq)

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
        self.softmax_scaler = softmax_scaler if softmax_scaler is not None else 1/np.sqrt(self.binding_dim)

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
        self.last_attn_scores = normalized_binding_matrix

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

# The implementation above learns the "relational templates" as parameters of the model directly along with the symbols
# and performs attention 'manually'.
# The alternative implementation below only learns the symbols and uses MultiHeadAttention to perform symbol retrieval.
# The relational templates are then given by a linear projection of the symbols.
class MultiHeadSymbolRetrieval2(tf.keras.layers.Layer):
    def __init__(self, n_heads, n_symbols, symbol_dim, binding_dim=None, use_bias=False, symbol_initializer='random_normal', **kwargs):
        super(MultiHeadSymbolRetrieval2, self).__init__(**kwargs)

        self.n_heads = n_heads
        self.n_symbols = n_symbols
        self.symbol_dim = symbol_dim
        self.binding_dim = binding_dim if binding_dim is not None else symbol_dim
        self.use_bias = use_bias
        self.symbol_initializer = symbol_initializer

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
        self.symbol_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.n_heads, key_dim=self.binding_dim, use_bias=self.use_bias, name='symbolic_attention')

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        symbol_library = tf.tile(tf.expand_dims(self.symbols, axis=0), [batch_size, 1, 1])
        retrieved_symbols, symbol_attn_scores = self.symbol_mha(query=inputs, value=symbol_library, return_attention_scores=True)
        self.last_attn_scores = symbol_attn_scores

        return retrieved_symbols