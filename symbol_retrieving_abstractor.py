import tensorflow as tf
from transformer_modules import FeedForward
from attention import GlobalSelfAttention, RelationalAttention
from abstracters import RelationalAbstracterLayer

# TODO: decide on how to integrate this into code-base and name of module
# maybe just integrate into abstracters.py?

class SymbolRetrievingAbstractor(tf.keras.layers.Layer):
    """
    An implementation of the Abstractor 2.0 module.

    1) Retrieve symbols
    2) Relational cross-attention
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        dff,
        n_symbols,
        binding_dim,
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
        self.binding_dim = binding_dim
        self.rel_activation_function = rel_activation_function
        self.use_self_attn = use_self_attn
        self.dropout_rate = dropout_rate

        # NOTE: we choose symbol_dim to be the same as d_model
        # this is required for residual connection to work
        # TODO think about whether this should be adjusted...

    def build(self, input_shape):

        _, self.sequence_length, self.d_model = input_shape

        self.symbol_retrieval = SymbolRetrieval(self.n_symbols, self.d_model, self.binding_dim)

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

class SymbolRetrieval(tf.keras.layers.Layer):
    def __init__(self, n_symbols, symbol_dim, binding_dim, symbol_initializer='random_normal'):
        super(SymbolRetrieval, self).__init__()

        self.n_symbols = n_symbols
        self.symbol_dim = symbol_dim
        self.binding_dim = binding_dim
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
        self.key_mapping = tf.keras.layers.Dense(self.binding_dim, use_bias=False)

    def call(self, inputs):
        input_keys = self.key_mapping(inputs)
        binding_matrix = tf.einsum('bik,jk->bij', input_keys, self.binding)
        normalized_binding_matrix = tf.nn.softmax(binding_matrix, axis=-1)
        retrieved_symbols = tf.einsum('bij,jk->bik', normalized_binding_matrix, self.symbols)
        return retrieved_symbols
