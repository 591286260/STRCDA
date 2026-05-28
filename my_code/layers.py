from initializations import *
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

# Dictionary for assigning globally unique IDs to layers.
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Assign a unique ID to each layer name.

    This function is used to avoid name conflicts when multiple layers
    of the same type are created in the computational graph.
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Apply dropout to a sparse tensor.

    Parameters
    ----------
    x : tf.SparseTensor
        Input sparse tensor.
    keep_prob : float
        Probability of keeping each non-zero element.
    num_nonzero_elems : int
        Number of non-zero elements in the sparse tensor.

    Returns
    -------
    tf.SparseTensor
        Sparse tensor after dropout.
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)

    # Retain only the selected non-zero entries.
    pre_out = tf.sparse_retain(x, dropout_mask)

    # Rescale the retained values to preserve the expected feature magnitude.
    return pre_out * (1. / keep_prob)


class Layer(object):
    """Base class for all neural network layers.

    Attributes
    ----------
    name : str
        Variable scope name of the layer.
    vars : dict
        Trainable variables used by the layer.
    logging : bool
        Whether to enable additional logging.
    issparse : bool
        Whether the layer receives sparse input.
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))

        self.name = name
        self.vars = {}
        self.logging = kwargs.get('logging', False)
        self.issparse = False

    def _call(self, inputs):
        """Define the forward computation of the layer.

        This method should be overridden by subclasses.
        """
        return inputs

    def __call__(self, inputs):
        """Wrapper for executing the layer within a TensorFlow name scope."""
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Graph convolution layer for dense node features.

    This layer performs feature transformation followed by neighborhood
    aggregation using the normalized adjacency matrix.
    """

    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name="weights"
            )

        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs

        # Apply dropout to dense input features.
        x = tf.nn.dropout(x, 1 - self.dropout)

        # Linear feature transformation.
        x = tf.matmul(x, self.vars['weights'])

        # Aggregate transformed features from neighboring nodes.
        x = tf.sparse_tensor_dense_matmul(self.adj, x)

        # Apply nonlinear activation.
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse input features.

    This layer is typically used as the first GCN layer when the input
    feature matrix is sparse.
    """

    def __init__(self, input_dim, output_dim, adj, features_nonzero,
                 dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name="weights"
            )

        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs

        # Apply dropout to sparse input features.
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)

        # Transform sparse features into dense hidden representations.
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])

        # Perform graph-based neighborhood aggregation.
        x = tf.sparse_tensor_dense_matmul(self.adj, x)

        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Inner-product decoder for link prediction.

    The decoder reconstructs pairwise association scores by computing
    the inner product between node embeddings.
    """

    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        # Apply dropout to node embeddings before reconstruction.
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)

        # Compute pairwise inner products between all node embeddings.
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)

        # Flatten the reconstructed adjacency matrix into a vector.
        x = tf.reshape(x, [-1])

        # Convert reconstruction scores into probabilities.
        outputs = self.act(x)
        return outputs


class ClusteringLayer(Layer):
    """Clustering layer based on Student's t-distribution.

    This layer computes the soft assignment probability of each node
    embedding to each cluster center.
    """

    def __init__(self, input_dim, n_clusters=7, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

        # Trainable cluster centers.
        self.vars['clusters'] = weight_variable_glorot(
            self.n_clusters, input_dim, name="cluster_weight"
        )

        # If pre-trained cluster centers are available, they can be assigned here.
        # self.vars['clusters'].assign(self.initial_weights)

    def _call(self, inputs):
        # Compute the distance between each embedding and each cluster center.
        q = tf.constant(1.0) / (
            tf.constant(1.0) +
            tf.reduce_sum(
                tf.square(tf.expand_dims(inputs, axis=1) - self.vars['clusters']),
                axis=2
            ) / tf.constant(self.alpha)
        )

        # Convert distances into soft assignment probabilities.
        q = tf.pow(q, tf.constant((self.alpha + 1.0) / 2.0))

        # Normalize cluster assignments for each sample.
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))

        return q
```
