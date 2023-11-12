import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class PrediNet(tf.keras.layers.Layer):
    """PrediNet layer (Shanahan et al. 2020)"""

    def __init__(self, key_dim, n_heads, n_relations, add_temp_tag=False):
        """create PrediNet layer.

        Parameters
        ----------
        key_dim : int
            key dimension
        n_heads : int
            number of heads
        n_relations : int
            number of relations
        add_temp_tag : bool, optional
            whether to add temporal tag to object representations, by default False
        """

        super(PrediNet, self).__init__()
        self.key_dim = key_dim
        self.n_heads = n_heads
        self.n_relations = n_relations
        self.add_temp_tag = add_temp_tag

    def build(self, input_shape):
        _, self.n_objs, obj_dim = input_shape

        self.obj_dim = obj_dim
        self.obj_tagged_dim = self.obj_dim + 1

        self.W_k = layers.Dense(self.key_dim, use_bias=False)
        self.W_q1 = layers.Dense(self.n_heads * self.key_dim, use_bias=False)
        self.W_q2 = layers.Dense(self.n_heads * self.key_dim, use_bias=False)
        self.W_s = layers.Dense(self.n_relations, use_bias=False)

        self.relu = layers.ReLU()
        self.softmax = layers.Softmax(axis=1)
        self.flatten = layers.Flatten()

        # create temporal tag
        if self.add_temp_tag:
            self.temp_tag = tf.convert_to_tensor(np.arange(self.n_objs), dtype=tf.float32)
            self.temp_tag = tf.expand_dims(self.temp_tag, axis=0)
            self.temp_tag = tf.expand_dims(self.temp_tag, axis=2)


    def call(self, obj_seq):

        # append temporal tag to all z
        if self.add_temp_tag:
            temp_tag = tf.tile(self.temp_tag, multiples=[tf.shape(obj_seq)[0], 1, 1])
            obj_seq = tf.concat([obj_seq, temp_tag], axis=2)

        # Get keys for all objects in sequence
        K = self.W_k(obj_seq)

        # get queries for objects 1 and 2
        obj_seq_flat = self.flatten(obj_seq)
        Q1 = self.W_q1(obj_seq_flat)
        Q2 = self.W_q2(obj_seq_flat)

        # reshape queries to separate heads
        Q1_reshaped = tf.reshape(Q1, shape=(-1, self.n_heads, self.key_dim))
        Q2_reshaped = tf.reshape(Q2, shape=(-1, self.n_heads, self.key_dim))

        # extract attended objects
        E1 = (self.softmax(tf.reduce_sum(Q1_reshaped[:, tf.newaxis, :, :] * K[:, :, tf.newaxis, :], axis=3))
             [:, :, :, tf.newaxis] * obj_seq[:, :, tf.newaxis, :])
        E1 = tf.reduce_sum(E1, axis=1)
        E2 = (self.softmax(tf.reduce_sum(Q2_reshaped[:, tf.newaxis, :, :] * K[:, :, tf.newaxis, :], axis=3))
              [:, :, :, tf.newaxis] * obj_seq[:, :, tf.newaxis, :])
        E2 = tf.reduce_sum(E2, axis=1)

        # compute relation vector
        D = self.W_s(E1) - self.W_s(E2)

        # add temporal position tag
        if self.add_temp_tag:
            D = tf.concat([D, E1[:, :, -1][:, :, tf.newaxis], E2[:, :, -1][:, :, tf.newaxis]], axis=2)

        R = self.flatten(D) # concatenate heads

        return R