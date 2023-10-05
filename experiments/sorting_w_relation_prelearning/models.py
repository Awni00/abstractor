import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import sys; sys.path.append('..'); sys.path.append('../..')
from transformer_modules import Encoder, Decoder, AddPositionalEmbedding
from abstracters import SimpleAbstractor

# default kwargs
simple_abstractor_kwargs = dict(
    num_layers=1, num_heads=3, dff=32,
    use_pos_embedding=False, mha_activation_type='sigmoid')

rel_model_kwargs = dict(embedding_dim=64, name='abstractor_order_rel')

decoder_kwargs = dict(num_layers=2, num_heads=4, dff=32, dropout_rate=0.1)

class AbstractorOrderRelation(tf.keras.Model):
    def __init__(self, embedding_dim, sigmoid_reg_lamda=10., name=None):
        super().__init__(name=name)
        self.embedding_dim = embedding_dim
        self.sigmoid_reg_lamda = sigmoid_reg_lamda
    
    def build(self, input_shape):
        self.embedder = layers.TimeDistributed(layers.Dense(self.embedding_dim))
        self.abstractor = SimpleAbstractor(**simple_abstractor_kwargs)
        self.dense = layers.Dense(1, activation='sigmoid', 
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.sigmoid_reg_lamda))

    def call(self, inputs):
        x = self.embedder(inputs)
        x = self.abstractor(x)
        x = x[:, 0]
        x = self.dense(x)

        return x 


def reload_rel_model(weights_path, object_dim, kwargs):
    model = AbstractorOrderRelation(**kwargs)
    model(np.random.random(size=(128, 2, object_dim)));
    model.load_weights(weights_path)
    return model

class AutoregressiveSimpleAbstractor(tf.keras.Model):
    def __init__(self, embedding_dim, seqs_length, decoder_kwargs, name=None):
      super().__init__(name=name)
  
      self.embedding_dim = embedding_dim
      self.target_vocab = seqs_length + 1
      self.output_dim = seqs_length
      self.decoder_kwargs = decoder_kwargs
  
    def build(self, input_shape):
        self.source_embedder = layers.TimeDistributed(layers.Dense(self.embedding_dim))
        self.abstractor = SimpleAbstractor(**simple_abstractor_kwargs)

        self.target_embedder = layers.Embedding(self.target_vocab, self.embedding_dim, name='target_embedder')
        self.pos_embedding_adder_target = AddPositionalEmbedding(name='add_pos_embedding_target')

        self.decoder = Decoder(**self.decoder_kwargs, name='decoder')
        self.final_layer = layers.Dense(self.output_dim, name='final_layer')

    def call(self, inputs):
        source, target  = inputs

        x = self.source_embedder(source)

        abstracted_context = self.abstractor(x)

        target_embedding = self.target_embedder(target)
        target_embedding = self.pos_embedding_adder_target(target_embedding)

        x = self.decoder(x=target_embedding, context=abstracted_context)

        logits = self.final_layer(x)

        try:
          # Drop the keras mask, so it doesn't scale the losses/metrics.
          # b/250038731
          del logits._keras_mask
        except AttributeError:
          pass

        return logits

def reload_argsort_model(weights_path, object_dim, seqs_length, kwargs):
    argsort_model = AutoregressiveSimpleAbstractor(embedding_dim=rel_model_kwargs['embedding_dim'], 
        seqs_length=seqs_length, decoder_kwargs=decoder_kwargs)
    
    target_example = np.random.randint(seqs_length, size=(128, seqs_length))
    source_example = np.random.random(size=(128, seqs_length, object_dim))
    argsort_model((source_example, target_example));

    argsort_model.load_weights(weights_path)

    return argsort_model

def initialize_with_rel_model(argsort_model, rel_model):
    '''initializez SimpleAbstractor argsort model with relation model'''

    # get weights from rel_model's abstractor
    rel_model_abstractor_weights = rel_model.abstractor.weights
    # set symbols to be those from the autoregressive model (they don't match in size)
    rel_model_abstractor_weights[0] = argsort_model.abstractor.weights[0]

    argsort_model.abstractor.set_weights(rel_model_abstractor_weights)
