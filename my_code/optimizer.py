import tensorflow.compat.v1 as tf
from sklearn.neighbors import NearestNeighbors
flags = tf.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, reconstructions1, adj, reconstructions2, features, model, global_step, learning_rate, pos_weight, norm, lambda_l1):

        self.cost1 = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions1, targets=adj, pos_weight=pos_weight))
        self.cost3 = tf.losses.mean_squared_error(reconstructions2,features)
        # 使用 tf.get_collection 来获取所有可训练变量
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # 计算 L1 正则化损失
        l1_loss = 0
        l2_loss = 0
        for var in trainable_vars:
            if 'weights' in var.name:  # 对权重变量应用正则化
                l1_loss += tf.reduce_sum(tf.abs(var))  # L1 正则化是权重绝对值之和
                l2_loss += tf.nn.l2_loss(tf.cast(var, tf.float32))

        self.cost = self.cost1 + self.cost3 + lambda_l1 * (l1_loss+l2_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost, global_step=global_step)


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost1 = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost1
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                tf.square(tf.exp(model.z_log_std)), 1))

        # distance = tf.reduce_sum(tf.square(tf.subtract(model.z_mean, tf.expand_dims(model.z_mean, 1))), axis=2)
        # d, top_k_indices = tf.nn.top_k(tf.negative(distance), k=100)
        # self.cost2 = tf.reduce_mean(tf.negative(d))

        self.cost1 -= self.kl
        # self.cost = self.cost1+self.cost2
        self.cost = self.cost1
        self.opt_op = self.optimizer.minimize(self.cost1)# 优化重构损失
        self.opt_op2 = self.optimizer.minimize(self.cost)# 优化总损失（包括 KL 散度）
        self.grads_vars = self.optimizer.compute_gradients(self.cost1)# 计算梯度
        # 准确率
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
