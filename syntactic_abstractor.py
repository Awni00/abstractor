"""
This module implements what we called a 'Syntactic Abstractor'. This is an experiment that we ran which didn't make it into the paper.

This module retrieves a symbol for each input via "symbolic attention" then performs self-attention on the retrieved symbols.
I.e., it is a mix between an Abstractor and an Encoder. There is no relational cross-attention, but there are 'symbols'.
This didn't work especially well in our experiments.
"""

import tensorflow as tf
from transformer_modules import EncoderLayer, AddPositionalEmbedding
from symbol_retrieving_abstractor import MultiHeadSymbolRetriever, MultiHeadSymbolRetrieval2

class SyntacticAbstractor(tf.keras.layers.Layer):
    """
    An implementation of the SyntacticAbstractor Abstractor module.

    1) Retrieve symbols
    2) Self-attention
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
        symbol_retriever_type=1, # there are two implementations; which one to use.
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
        symbol_retriever_type : int, optional
            type of symbol retriever, by default 1.
        dropout_rate : float, optional
            dropout rate, by default 0.1
        **kwargs : dict
            kwargs for parent Layer class
        """

        super(SyntacticAbstractor, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.n_symbols = n_symbols
        self.symbol_n_heads = symbol_n_heads
        self.symbol_binding_dim = symbol_binding_dim
        self.should_add_pos_embedding = add_pos_embedding
        self.symbol_retriever_type = symbol_retriever_type
        self.dropout_rate = dropout_rate

        # NOTE: we choose symbol_dim to be the same as d_model
        # this is required for residual connection to work

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

        self.encoder_layers = [
            EncoderLayer(d_model=self.d_model, num_heads=self.num_heads,
                dff=self.dff, dropout_rate=self.dropout_rate)
            for _ in range(self.num_layers)]

        self.last_attn_scores = None

    def call(self, inputs):

        symbol_seq = self.symbol_retrieval(inputs) # retrieve symbols
        if self.should_add_pos_embedding:
            symbol_seq = self.add_pos_embedding(symbol_seq)

        symbol_seq = self.dropout(symbol_seq)

        for i in range(self.num_layers):
            symbol_seq = self.encoder_layers[i](symbol_seq)

        return symbol_seq
