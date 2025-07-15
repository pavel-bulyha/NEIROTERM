# utils/weightedBCE.py

"""
Factory for weighted binary cross-entropy loss.
Automatically adapts to class imbalance by assigning
a higher penalty for errors on the rare (positive) class.

Functions:
    weighted_BCE(pos_weight) → loss_fn
"""

import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore

def weighted_BCE(pos_weight: float):
    """
    Returns a loss function for weighted BCE.

    Args:
        pos_weight: weight for the positive class (label=1).

    Returns:
        loss_fn: function loss(y_true, y_pred), where y_pred are probabilities.
    """
    def loss_fn(y_true, y_pred):
        # y_true: {0,1}, y_pred: [0,1]
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)

        # Weighted BCE: 
        # - pos_weight * y_true * log(y_pred)
        # -         (1-y_true) * log(1 - y_pred)
        loss_pos = - pos_weight * y_true * tf.math.log(y_pred) # type: ignore
        loss_neg = - (1 - y_true) * tf.math.log(1 - y_pred) # type: ignore
        return tf.reduce_mean(loss_pos + loss_neg)

    return loss_fn
