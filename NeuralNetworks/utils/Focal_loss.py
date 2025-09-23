# utils/focal_loss.py

"""
Factory for focal loss.
Returns a focal loss function to focus training on hard examples.

Functions:
    focal_loss(alpha, gamma) → loss_fn
"""

import tensorflow as tf
from tensorflow.keras import backend as K  # type: ignore

def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """
    Returns a focal loss function to focus training on hard examples.

    Args:
        alpha:   balancing factor in [0,1] for positive class.
        gamma:   focusing parameter ≥ 0; higher → focuses more on hard examples.

    Returns:
        loss_fn: function loss(y_true, y_pred), where y_pred are probabilities.
    """
    def loss_fn(y_true, y_pred):
        # y_true: {0,1}, y_pred: [0,1]
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)

        # p_t = y_pred for true class, else (1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred) # type: ignore
        # alpha_t = alpha for positive, (1-alpha) for negative
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha) # type: ignore
        # cross-entropy term
        ce = - tf.math.log(p_t)  # type: ignore
        # focal modulator
        mod = tf.pow(1 - p_t, gamma)

        loss = alpha_t * mod * ce
        return tf.reduce_mean(loss)

    return loss_fn
