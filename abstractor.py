import tensorflow as tf
from tensorflow.keras import layers
from multi_head_relation import MultiHeadRelation
from transformer_modules import GlobalSelfAttention, create_positional_encoding, FeedForward

# TODO: add feedforward layers after message-passing (like RelationalAbstracter)

class Abstractor(tf.keras.layers.Layer):
    def __init__(self,
        num_layers,
        rel_dim,
        dff,
        symbol_dim=None,
        use_learned_symbols=True,
        proj_dim=None,
        symmetric_rels=False,
        encoder_kwargs=None,
        rel_activation_type='softmax',
        use_self_attn=False,
        use_layer_norm=False,
        dropout_rate=0.,
        name=None):
        """
        create an Abstractor layer.

        Models relations between objects via a relation tensor (from MultiHeadRelation),
        and performs message-passing on a set of input-independent symbolic parameters
        based on the relation tensor ("(relational) symbolic message-passing").

        Unlike RelationalAbstractor, this layer does not use tensorflow's MultiHeadAttention,
        instead implementing 'symbolic message-passing' directly from scratch.

        Parameters
        ----------
        num_layers : int
            number of Abstractor layers (i.e., number of symbolic message-passing operations)
        rel_dim : int
            dimension of relations. applies to MultiHeadRelation in each layer.
        symbol_dim : int, optional
            dimension of symbols, by default None
        use_learned_symbols: bool, optional
            whether to use learned symbols or nonparametric sinusoidal symbols.
            If learned, there will be a limit to the input length. by default True
        proj_dim : int, optional
            dimension of projections in MultiHeadRelation layers, by default None
        symmetric_rels : bool, optional
            whether to model relations as symmetric or not in MultiHeadRelation layers, by default False
        encoder_kwargs : dict, optional
            kwargs of Dense encoders in MultiHeadRelation layers, by default None
        rel_activation_type : str, optional
            name of activation function to use on relation tensor, by default 'softmax'
        use_self_attn : bool, optional
            whether or not to use self-attention, by default False
        dropout_rate : float, optional
            dropout rate, by default 0.
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)

        self.num_layers = num_layers
        self.rel_dim = rel_dim
        self.dff = dff
        self.proj_dim = proj_dim
        self.symmetric_rels = symmetric_rels
        self.encoder_kwargs = encoder_kwargs
        self.symbol_dim = symbol_dim
        self.use_learned_symbols = use_learned_symbols
        self.rel_activation_type = rel_activation_type
        self.use_self_attn = use_self_attn
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.max_length = 1024 # TODO: make this configurable?

    def build(self, input_shape):

        _, self.sequence_length, self.object_dim = input_shape

        self.max_length = max(self.sequence_length, self.max_length)

        # symbol_dim is not given, use same dimension as objects
        if self.symbol_dim is None:
            self.symbol_dim = self.object_dim

        if self.proj_dim is None:
            self.proj_dim = self.object_dim

        # define the input-independent symbolic input vector sequence
        if self.use_learned_symbols:
            normal_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            self.symbol_sequence = tf.Variable(
                normal_initializer(shape=(self.sequence_length, self.symbol_dim)),
                name='symbols', trainable=True)
        else:
            # create non-parametric sinusoidal symbols
            self.symbol_sequence = create_positional_encoding(length=self.max_length, depth=self.symbol_dim)

        if self.use_self_attn:
            self.self_attention_layers = [GlobalSelfAttention(
                num_heads=self.rel_dim,
                key_dim=self.proj_dim,
                activation_type='softmax',
                dropout=self.dropout_rate) for _ in range(self.num_layers)]

        # MultiHeadRelation layer for each layer of Abstractor
        self.multi_head_relation_layers = [MultiHeadRelation(
            rel_dim=self.rel_dim, proj_dim=self.proj_dim,
            symmetric=self.symmetric_rels, dense_kwargs=self.encoder_kwargs)
            for _ in range(self.num_layers)]

        if self.rel_activation_type == 'softmax':
            self.rel_activation = tf.keras.layers.Softmax(axis=-2)
        else:
            self.rel_activation = tf.keras.layers.Activation(self.rel_activation_type)

        #W_o^h; output projection layers for each relation dim
        self.symbol_proj_layers = [[layers.Dense(self.symbol_dim // self.rel_dim) for _ in range(self.rel_dim)] for _ in range(self.num_layers)]

        # feedforward layers
        self.ff_layers = [FeedForward(self.symbol_dim, self.dff) for _ in range(self.num_layers)]

        if self.use_layer_norm:
            self.layer_norms = [layers.LayerNormalization()]*self.num_layers

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)


    def call(self, inputs):

        m = tf.shape(inputs)[1]
        symbol_sequence = self.symbol_sequence[:m, :]

        for i in range(self.num_layers):

            # get relation tensor via MultiHeadRelation layer
            rel_tensor = self.multi_head_relation_layers[i](inputs) # shape: [b, m, m, d_r]

            # apply activation to relation tensor (e.g.: softmax)
            rel_tensor = self.rel_activation(rel_tensor)

            # perform symbolic message-passing based on relation tensor
            # A_bijr = sum_k R_bikr S_bkj (A = S.T @ R)
            if i == 0: # on first iteration, symbol equence is untransformed of shape [m, d_s]
                abstract_symbol_seq = tf.einsum('bikr,kj->bijr', rel_tensor, symbol_sequence) # shape: [b, m, d_s, d_r]
            else: # on next iterations, symbol sequence is transformed with shape [b, m, d_s]
                abstract_symbol_seq = tf.einsum('bikr,bkj->bijr', rel_tensor, abstract_symbol_seq) # shape: [b, m, d_s, d_r]

            # project and concatenate
            abstract_symbol_seq = tf.concat([self.symbol_proj_layers[i][r](abstract_symbol_seq[:, :, :, r]) for r in range(self.rel_dim)], axis=2) # shape: [b, m, d_s]

            # transform symbol sequence via dense layer to return to its original dimension
            abstract_symbol_seq = self.ff_layers[i](abstract_symbol_seq) # shape: [b, m, d_s]

            if self.use_layer_norm:
                abstract_symbol_seq = self.layer_norms[i](abstract_symbol_seq)

            # apply self-attention to symbol sequence
            if self.use_self_attn:
                # need to expand dims to add batch dim first
                abstract_symbol_seq = self.self_attention_layers[i](abstract_symbol_seq) # shape [b, m, d_s]

            # dropout
            abstract_symbol_seq = self.dropout(abstract_symbol_seq)

        return abstract_symbol_seq


    def get_config(self):
        config = super(Abstractor, self).get_config()

        config.update(
            {
                'num_layers': self.num_layers,
                'rel_dim': self.rel_dim,
                'proj_dim': self.proj_dim,
                'symmetric_rels': self.symmetric_rels,
                'encoder_kwargs': self.encoder_kwargs,
                'symbol_dim': self.symbol_dim,
                'rel_activation_type': self.rel_activation_type,
                'dropout_rate': self.dropout_rate
            })

        return config