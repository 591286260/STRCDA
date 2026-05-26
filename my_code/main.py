#dgae coil20 图片实验
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

from preprocessing import (construct_feed_dict, mask_test_edges,
                           preprocess_graph, sparse_to_tuple)
# import warnings filter
from warnings import simplefilter

import scipy.io as scio
import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""




# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs',500, 'Number of epochs to train.')#500
flags.DEFINE_integer('epochs2',200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('n_clusters',20, 'Number of cluster.')


flags.DEFINE_integer('k',12,'Number of k.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
###main,model，flp改维度
path = '../dataset/1-CircR2Disease(561-100-607)/'#661
# path = '../dataset/2-CircAtlas 2.0(708-117-775)/'#825
# path = '../dataset/3-Circ2Disease(234-60-254)/'#294
# path = '../dataset/4-CircRNADisease(286-48-304)/'#334

num_circ = 561
num_dis = 100
con = 607


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


for i in range(1):
    AM = np.loadtxt(path+'AM.csv', delimiter=',')
    CS = np.loadtxt(path+'RNA_sim.csv', delimiter=',')
    DS = np.loadtxt(path+'Disease_sim.csv', delimiter=',')
    is_rwr = True
    if is_rwr == True:
        feature = np.loadtxt(path+"feature.csv", delimiter=',')
    else:
        feature = get_feature(CS, AM, DS)

    print('特征shape', feature.shape)

    labels = []#circRNA为1，疾病为0
    for i in range(num_circ):
        labels.append(1)
    for i in range(num_dis):
        labels.append(0)
    features = sparse.lil_matrix(feature)
    adj = kneighbors_graph(feature, FLAGS.k, mode='connectivity', include_self=True)
    featuresa = features.toarray()
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step=global_step,decay_rate=0.98,staircase=True,decay_steps=50)
    s=0
    indices = 0

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)

    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    feed_dict = construct_feed_dict(adj_norm, features, placeholders)
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # l1 = sys.argv[1]
    # l1 = float(l1)
    # print("l1", l1)
    aaa=np.zeros((561,100))
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(reconstructions1=model.reconstructions1,
                              adj=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj'],
                                                                          validate_indices=False), [-1]),
                              reconstructions2=model.reconstructions2,
                              features=tf.reshape(tf.sparse_tensor_to_dense(placeholders['features'],
                                                                          validate_indices=False), [num_circ+num_dis, num_circ+num_dis]),
                              model=model,
                              pos_weight=pos_weight,
                              norm=norm,
                              global_step=global_step,
                              learning_rate=learning_rate,
                              lambda_l1=0.0001#0.00001
                              )

        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    sess.run(tf.global_variables_initializer())

    for epoch in range(FLAGS.epochs):

        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        outs = sess.run([opt.opt_op, opt.cost, opt.cost, opt.cost1, global_step, learning_rate], feed_dict=feed_dict)
        cost_v = outs[1]
        costkl1_v = outs[2]
        costkl_v = outs[3]
        step_v = outs[4]
        lr = outs[5]

        if epoch % 50 == 0:
            print("Epoch:", '%04d' % (step_v + 1), "train_loss=", "{:.5f}".format(cost_v),)

    emb = sess.run(model.embeddings, feed_dict=feed_dict)

    rec_adj = sess.run(model.reconstructions1, feed_dict=feed_dict)  # 436921*1
    rec_fea = sess.run(model.reconstructions2, feed_dict=feed_dict)  # 661*661

    # print(emb)
    # print(rec_adj)
    # print(rec_fea)

    if is_rwr == True:
        # np.savetxt(path + "0-data_rec_adj_rwr", rec_adj)
        # np.savetxt(path + "0-data_rec_fea_rwr", rec_fea)
        # np.savetxt(path + "0-emb_train_rwr", emb)
        # print("rwr-保存完成")
        #临时保存
        np.savetxt("0-data_rec_adj_rwr", rec_adj)
        np.savetxt("0-data_rec_fea_rwr", rec_fea)
        np.savetxt("0-emb_train_rwr", emb)
    else:
        np.savetxt(path + "1-data_rec_adj", rec_adj)
        np.savetxt(path + "1-data_rec_fea", rec_fea)
        np.savetxt(path + "1-emb_train", emb)
        print("no-rwr-保存完成")
        # 临时保存
        # np.savetxt("1-data_rec_adj", rec_adj)
        # np.savetxt("1-data_rec_fea", rec_fea)
        # np.savetxt("1-emb_train", emb)

