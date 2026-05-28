import tensorflow.compat.v1 as tf
from sklearn.neighbors import NearestNeighbors


flags = tf.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    """
    Optimizer for Graph Autoencoder (GCN-AE).

    This optimizer jointly minimizes:
    1. Adjacency matrix reconstruction loss
    2. Feature matrix reconstruction loss
    3. L1 and L2 regularization losses
    """

    def __init__(self,
                 reconstructions1,
                 adj,
                 reconstructions2,
                 features,
                 model,
                 global_step,
                 learning_rate,
                 pos_weight,
                 norm,
                 lambda_l1):

        # ====================================================
        # Adjacency reconstruction loss
        # Weighted cross-entropy is used to address class imbalance
        # ====================================================

        self.cost1 = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=reconstructions1,
                targets=adj,
                pos_weight=pos_weight
            )
        )

        # ====================================================
        # Feature reconstruction loss
        # Mean squared error between reconstructed and original features
        # ====================================================

        self.cost3 = tf.losses.mean_squared_error(
            reconstructions2,
            features
        )

        # ====================================================
        # Regularization losses
        # ====================================================

        # Retrieve all trainable variables
        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES
        )

        l1_loss = 0
        l2_loss = 0

        for var in trainable_vars:

            # Apply regularization only to weight matrices
            if 'weights' in var.name:

                # L1 regularization
                l1_loss += tf.reduce_sum(tf.abs(var))

                # L2 regularization
                l2_loss += tf.nn.l2_loss(
                    tf.cast(var, tf.float32)
                )

        # ====================================================
        # Total optimization objective
        # ====================================================

        self.cost = (
                self.cost1 +
                self.cost3 +
                lambda_l1 * (l1_loss + l2_loss)
        )

        # Adam optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        )

        # Training operation
        self.opt_op = self.optimizer.minimize(
            self.cost,
            global_step=global_step
        )


class OptimizerVAE(object):
    """
    Optimizer for Graph Variational Autoencoder (GCN-VAE).

    This optimizer minimizes:
    1. Graph reconstruction loss
    2. KL divergence regularization
    """

    def __init__(self,
                 preds,
                 labels,
                 model,
                 num_nodes,
                 pos_weight,
                 norm):

        preds_sub = preds
        labels_sub = labels

        # ====================================================
        # Reconstruction loss
        # ====================================================

        self.cost1 = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub,
                targets=labels_sub,
                pos_weight=pos_weight
            )
        )

        # Adam optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate
        )

        # ====================================================
        # KL divergence loss
        # Regularizes latent embedding distribution
        # ====================================================

        self.log_lik = self.cost1

        self.kl = (
                (0.5 / num_nodes) *
                tf.reduce_mean(
                    tf.reduce_sum(
                        1 +
                        2 * model.z_log_std -
                        tf.square(model.z_mean) -
                        tf.square(tf.exp(model.z_log_std)),
                        1
                    )
                )
        )

        # Final VAE loss
        self.cost1 -= self.kl

        self.cost = self.cost1

        # ====================================================
        # Optimization operations
        # ====================================================

        # Optimize reconstruction loss
        self.opt_op = self.optimizer.minimize(self.cost1)

        # Optimize total loss
        self.opt_op2 = self.optimizer.minimize(self.cost)

        # Compute gradients
        self.grads_vars = self.optimizer.compute_gradients(
            self.cost1
        )

        # ====================================================
        # Prediction accuracy
        # ====================================================

        self.correct_prediction = tf.equal(
            tf.cast(
                tf.greater_equal(
                    tf.sigmoid(preds_sub),
                    0.5
                ),
                tf.int32
            ),
            tf.cast(labels_sub, tf.int32)
        )

        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32)
        )
```
