# ============================================================
# STRCDA: Sparse Topological Representation Learning Framework
# for CircRNA-Disease Association Prediction
#
# This script is used to train the graph autoencoder model
# and generate latent node embeddings, reconstructed adjacency
# matrices, and reconstructed feature matrices.
#
# Main workflow:
# 1. Load similarity matrices and association network
# 2. Construct node features
# 3. Build k-nearest neighbor graph
# 4. Train graph autoencoder (GCN-AE / GCN-VAE)
# 5. Generate low-dimensional embeddings
# 6. Save reconstructed representations
#
# Dataset:
# CircR2Disease
#
# Framework:
# TensorFlow v1
# ============================================================

from __future__ import division, print_function

import os
import sys

import numpy
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from model import GCNModelAE, GCNModelVAE
from my_code.utilize import get_feature
from optimizer import OptimizerAE, OptimizerVAE

from sklearn.cluster import SpectralClustering

from preprocessing import (
    construct_feed_dict,
    mask_test_edges,
    preprocess_graph,
    sparse_to_tuple
)

from warnings import simplefilter

import scipy.io as scio
import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

# Ignore TensorFlow future warnings
simplefilter(action='ignore', category=FutureWarning)

# Train on CPU to avoid GPU memory limitations
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# ============================================================
# Hyperparameter Settings
# ============================================================

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of training epochs.')
flags.DEFINE_integer('epochs2', 200, 'Additional training epochs.')
flags.DEFINE_integer('hidden1', 256, 'Hidden units in encoder layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Hidden units in encoder layer 2.')
flags.DEFINE_integer('n_clusters', 20, 'Number of clustering centers.')

flags.DEFINE_integer('k', 12, 'Number of nearest neighbors.')
flags.DEFINE_float('weight_decay', 0., 'L2 regularization weight.')
flags.DEFINE_float('dropout', 0., 'Dropout rate.')
flags.DEFINE_string('model', 'gcn_ae', 'Graph autoencoder type.')
flags.DEFINE_string('dataset', 'cora', 'Dataset name.')
flags.DEFINE_integer('features', 1, 'Use node features or identity matrix.')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# ============================================================
# Dataset Configuration
# ============================================================

path = '../dataset/1-CircR2Disease(561-100-607)/'

num_circ = 561
num_dis = 100
con = 607


def target_distribution(q):
    """
    Compute target distribution for clustering refinement.

    Parameters
    ----------
    q : ndarray
        Soft cluster assignment matrix.

    Returns
    -------
    ndarray
        Refined target distribution.
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


for i in range(1):

    # ========================================================
    # Load association and similarity matrices
    # ========================================================

    AM = np.loadtxt(path + 'AM.csv', delimiter=',')
    CS = np.loadtxt(path + 'RNA_sim.csv', delimiter=',')
    DS = np.loadtxt(path + 'Disease_sim.csv', delimiter=',')

    # Whether to use precomputed RWR features
    is_rwr = True

    if is_rwr:
        feature = np.loadtxt(path + "feature.csv", delimiter=',')
    else:
        feature = get_feature(CS, AM, DS)

    print('Feature shape:', feature.shape)

    # ========================================================
    # Construct node labels
    # circRNA: 1
    # Disease: 0
    # ========================================================

    labels = []

    for i in range(num_circ):
        labels.append(1)

    for i in range(num_dis):
        labels.append(0)

    # Convert feature matrix to sparse format
    features = sparse.lil_matrix(feature)

    # Construct k-nearest neighbor graph
    adj = kneighbors_graph(
        feature,
        FLAGS.k,
        mode='connectivity',
        include_self=True
    )

    featuresa = features.toarray()

    # Learning rate decay strategy
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        global_step=global_step,
        decay_rate=0.98,
        staircase=True,
        decay_steps=50
    )

    if FLAGS.features == 0:
        # Use identity matrix when featureless setting is enabled
        features = sp.identity(features.shape[0])

    # ========================================================
    # Graph preprocessing
    # ========================================================

    adj_norm = preprocess_graph(adj)

    # TensorFlow placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # ========================================================
    # Build Graph Autoencoder Model
    # ========================================================

    model = None

    if model_str == 'gcn_ae':
        model = GCNModelAE(
            placeholders,
            num_features,
            features_nonzero
        )

    elif model_str == 'gcn_vae':
        model = GCNModelVAE(
            placeholders,
            num_features,
            num_nodes,
            features_nonzero
        )

    # ========================================================
    # Loss normalization
    # ========================================================

    pos_weight = float(
        adj.shape[0] * adj.shape[0] - adj.sum()
    ) / adj.sum()

    norm = adj.shape[0] * adj.shape[0] / float(
        (adj.shape[0] * adj.shape[0] - adj.sum()) * 2
    )

    feed_dict = construct_feed_dict(adj_norm, features, placeholders)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # ========================================================
    # Optimizer Definition
    # ========================================================

    with tf.name_scope('optimizer'):

        if model_str == 'gcn_ae':

            opt = OptimizerAE(
                reconstructions1=model.reconstructions1,
                adj=tf.reshape(
                    tf.sparse_tensor_to_dense(
                        placeholders['adj'],
                        validate_indices=False
                    ),
                    [-1]
                ),
                reconstructions2=model.reconstructions2,
                features=tf.reshape(
                    tf.sparse_tensor_to_dense(
                        placeholders['features'],
                        validate_indices=False
                    ),
                    [num_circ + num_dis, num_circ + num_dis]
                ),
                model=model,
                pos_weight=pos_weight,
                norm=norm,
                global_step=global_step,
                learning_rate=learning_rate,
                lambda_l1=0.0001
            )

        elif model_str == 'gcn_vae':

            opt = OptimizerVAE(
                preds=model.reconstructions,
                labels=tf.reshape(
                    tf.sparse_tensor_to_dense(
                        placeholders['adj'],
                        validate_indices=False
                    ),
                    [-1]
                ),
                model=model,
                num_nodes=num_nodes,
                pos_weight=pos_weight,
                norm=norm
            )

    sess.run(tf.global_variables_initializer())

    # ========================================================
    # Model Training
    # ========================================================

    for epoch in range(FLAGS.epochs):

        feed_dict.update({
            placeholders['dropout']: FLAGS.dropout
        })

        outs = sess.run(
            [
                opt.opt_op,
                opt.cost,
                opt.cost,
                opt.cost1,
                global_step,
                learning_rate
            ],
            feed_dict=feed_dict
        )

        cost_v = outs[1]
        step_v = outs[4]

        if epoch % 50 == 0:
            print(
                "Epoch:",
                '%04d' % (step_v + 1),
                "train_loss=",
                "{:.5f}".format(cost_v),
            )

    # ========================================================
    # Generate embeddings and reconstructed matrices
    # ========================================================

    emb = sess.run(model.embeddings, feed_dict=feed_dict)

    rec_adj = sess.run(
        model.reconstructions1,
        feed_dict=feed_dict
    )

    rec_fea = sess.run(
        model.reconstructions2,
        feed_dict=feed_dict
    )

    # ========================================================
    # Save results
    # ========================================================

    if is_rwr:

        np.savetxt("0-data_rec_adj_rwr", rec_adj)
        np.savetxt("0-data_rec_fea_rwr", rec_fea)
        np.savetxt("0-emb_train_rwr", emb)

    else:

        np.savetxt(path + "1-data_rec_adj", rec_adj)
        np.savetxt(path + "1-data_rec_fea", rec_fea)
        np.savetxt(path + "1-emb_train", emb)

        print("Results saved successfully.")
```
