import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadRelation(tf.keras.layers.Layer):
    def __init__(
        self,
        rel_dim,
        proj_dim=None,
        symmetric=False,
        dense_kwargs=None,
        name=None):
        """
        create a MultiHeadRelation module.

        Computes self-relations within a sequence of objects or cross-relations
        between two sequences of objects. Returns a relation tensor of shape
        [m1, m2, d_r] where m1 is the number of objects in the first sequence,
        m2 is the number of objects in the second sequence, and d_r is the
        dimension of the relation.

        Parameters
        ----------
        rel_dim : int
            dimension of the relation.
        proj_dim : int, optional
            dimension to which encoders will project before inner products
            are computed. by default None
        symmetric : bool, optional
            whether to model the relations as symmetric or not. 
            if the relations are symmetric, the left sequence will use the same
            encoders as the right sequence. otherwise, different encoders will
            be used. by default False
        dense_kwargs : dict, optional
            kwargs for Dense encoder layers. e.g. whether to use bias, etc.
            by default None
        name : str, optional
            name of layer, by default None
        """

        super().__init__(name=name)

        self.rel_dim = rel_dim
        self.proj_dim = proj_dim
        self.symmetric = symmetric
        if dense_kwargs is None:
            self.dense_kwargs = dict()
        elif isinstance(dense_kwargs, dict):
            self.dense_kwargs = dense_kwargs
        else:
            raise ValueError(f'`dense_kwargs` {dense_kwargs} invalid. must be dict.') 
    
    def build(self, input_shape):
        batch_dim, self.n_objects, self.object_dim = input_shape

        # if projection dimension not given, use object dimension
        if self.proj_dim is None:
            self.proj_dim = self.object_dim

        # if symmetric relations, left objects and right objects use the same linear encoders
        if self.symmetric:
            self.left_encoders = self.right_encoders = [layers.Dense(self.proj_dim, **self.dense_kwargs) for _ in range(self.rel_dim)]
        # otherwise create different encoders for each set of objects
        else:
            self.left_encoders =  [layers.Dense(self.proj_dim, **self.dense_kwargs) for _ in range(self.rel_dim)]
            self.right_encoders =  [layers.Dense(self.proj_dim, **self.dense_kwargs) for _ in range(self.rel_dim)]
    
    def call(self, left_objects, right_objects=None):

        # if `right_objects` not given, left_objects and right_objects are the same
        # i.e.: we are computing "self-relations"
        if right_objects is None:
            right_objects = left_objects

        relation_matrices = []

        # loop over encoders to get a relation matrix for each 'attribute' / 'filter'
        for (left_encoder, right_encoder) in zip(self.left_encoders, self.right_encoders):
            # encode each set of objects using their respective encoders
            encoded_left_objects = left_encoder(left_objects)
            encoded_right_objects = right_encoder(right_objects)

            # compute the relation matrix and append to list
            attr_rel_mat = tf.einsum('bmd,bnd->bmn', encoded_left_objects, encoded_right_objects)
    
            # attr_rel_mat = tf.matmul(encoded_left_objects, tf.transpose(encoded_right_objects, perm=(0,2,1)))
            relation_matrices.append(attr_rel_mat)

        # construct relation tensor from relation matrices
        relation_tensor = tf.stack(relation_matrices, axis=-1)

        return relation_tensor

    def get_config(self):
        config = super(MultiHeadRelation, self).get_config()
        config.update(
            {
                'rel_dim': self.rel_dim,
                'proj_dim': self.proj_dim,
                'symmetric': self.symmetric,
                'dense_kwargs': self.dense_kwargs
            }
            )

        return config