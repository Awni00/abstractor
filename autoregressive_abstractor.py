import tensorflow as tf
from tensorflow.keras import layers
from transformer_modules import Encoder, AddPositionalEmbedding
from multi_attention_decoder import MultiAttentionDecoder
from abstractor import Abstractor
from abstracters import RelationalAbstracter, SymbolicAbstracter

class AutoregressiveAbstractor(tf.keras.Model):
    """
    An implementation of an Abstractor-based Transformer module.

    This supports several architectures, including:
    a) X -> Abstractor -> Decoder -> Y
    b) X -> Encoder -> Abstractor -> Decoder -> Y
    c) X -> [Encoder, Abstractor] -> Decoder -> Y
    d) X -> Encoder -> Abstractor; [Encoder, Abstractor] -> Decoder -> Y
    """
    def __init__(self,
            encoder_kwargs,
            abstractor_kwargs,
            decoder_kwargs,
            input_vocab,
            target_vocab,
            embedding_dim,
            output_dim,
            abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
            abstractor_on='input', # 'input' or 'encoder'
            decoder_on='abstractor', # 'abstractor' or 'encoder-abstractor'
            name=None):
        """Creates an autoregressive Abstractor model.

        Parameters
        ----------
        encoder_kwargs : dict
            kwargs for the Encoder module. Can be set to None if architecture does not use an encoder.
        abstractor_kwargs : dict
            kwargs for the Abstractor model. Should match `abstractor_type`
        decoder_kwargs : dict
            kwargs for Decoder module.
        input_vocab : int or 'vector'
            if input is tokens, the size of vocabulary as an int.
            if input is vectors, the string 'vector'. used to create embedder.
        target_vocab : int or 'vector'
            if input is tokens, the size of vocabulary as an int.
            if input is vectors, the string 'vector'. used to create embedder.
        embedding_dim : int or tuple[int]
            dimension of embedding (input will be transformed to this dimension).
        output_dim : int
            dimension of final output. e.g.: # of classes.
        abstractor_type : 'abstractor', 'relational', or 'symbolic', optional
            The type of Abstractor to use, by default 'relational'
        abstractor_on: 'input' or 'encoder'
            what the abstractor should take as input.
        decoder_on: 'abstractor' or 'encoder-abstractor'
            what should form the decoder's 'context'.
            if 'abstractor' the context is the output of the abstractor.
            if 'encoder-abstractor' the context is the concatenation of the outputs of the encoder and decoder. 
        """

        super().__init__(name=name)

        # set params
        self.relation_on = abstractor_on
        self.decoder_on = decoder_on
        self.abstractor_type = abstractor_type

        # if relation is computed on inputs and the decoder attends only to the abstractor,
        # there is no need for an encoder
        if (abstractor_on, decoder_on) == ('input', 'abstractor'):
            self.use_encoder = False
            print(f'NOTE: no encoder will be used since relation_on={abstractor_on} and decoder_on={decoder_on}')
        else:
            self.use_encoder = True

        # set up source and target embedders
        if isinstance(input_vocab, int):
            self.source_embedder = layers.Embedding(input_vocab, embedding_dim, name='source_embedder')
        elif input_vocab == 'vector':
            self.source_embedder = layers.Dense(embedding_dim, name='source_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        if isinstance(target_vocab, int):
            self.target_embedder = layers.Embedding(target_vocab, embedding_dim, name='target_embedder')
        elif target_vocab == 'vector':
            self.target_embedder = layers.Dense(embedding_dim, name='target_embedder')
        else:
            raise ValueError(
                "`input_vocab` must be an integer if the input sequence is token-valued or "
                "'vector' if the input sequence is vector-valued.")

        self.pos_embedding_adder_input = AddPositionalEmbedding(name='add_pos_embedding_input')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        # initialize layers
        if self.use_encoder:
            self.encoder = Encoder(**encoder_kwargs, name='encoder')

        # initialize the abstractor based on requested type
        if abstractor_type == 'abstractor':
            self.abstractor = Abstractor(**abstractor_kwargs, name='abstractor')
        elif abstractor_type == 'relational':
            self.abstractor = RelationalAbstracter(**abstractor_kwargs, name='abstractor')
        elif abstractor_type == 'symbolic':
            self.abstractor = SymbolicAbstracter(**abstractor_kwargs, name='abstractor')
        else:
            raise ValueError(f'unexpected `abstracter_type` argument {abstractor_type}')

        # initialize decoder
        self.decoder = MultiAttentionDecoder(**decoder_kwargs, name='decoder')

        # initialize final prediction layer
        self.final_layer = layers.Dense(output_dim, name='final_layer')


    def call(self, inputs):
        source, target = inputs # get source and target from inputs

        # embed source and add positional embedding
        source = self.source_embedder(source)
        source = self.pos_embedding_adder_input(source)

        # pass input to Encoder
        if self.use_encoder:
            encoder_context = self.encoder(source)

        # compute abstracted context (either directly on embedded input or on encoder output)
        if self.relation_on == 'input':
            abstracted_context = self.abstractor(source)
        elif self.relation_on == 'encoder':
            abstracted_context = self.abstractor(encoder_context)
        else:
            raise ValueError()

        # embed target and add positional embedding
        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        # decode context (either abstractor only or concatenation of encoder and abstractor outputs)
        if self.decoder_on == 'abstractor':
            decoder_inputs = [target_embedding, abstracted_context]
        elif self.decoder_on == 'encoder-abstractor':
            decoder_inputs = [target_embedding, encoder_context, abstracted_context]
        else:
            raise ValueError()

        x = self.decoder(decoder_inputs)

        # produce final prediction
        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics. b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits
