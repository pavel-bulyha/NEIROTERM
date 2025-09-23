# utils/balanced_accuracy.py

"""
Factory for balanced accuracy loss.
Returns a differentiable proxy loss for balanced accuracy.

Balanced Accuracy = 0.5 * (TP / P + TN / N)
Loss = 1 - Balanced Accuracy, using soft counts (probabilities).

Functions:
    balanced_accuracy_loss() → loss_fn
"""

import tensorflow as tf
from tensorflow.keras import backend as K  # type: ignore

def balanced_accuracy_loss():
    """
    Returns a differentiable proxy loss for balanced accuracy.

    Balanced Accuracy = 0.5 * (TP / P + TN / N)
    Loss = 1 - Balanced Accuracy, using soft counts (probabilities).
    """
    def loss_fn(y_true, y_pred):
        # y_true: {0,1}, y_pred: [0,1]
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)

        # Positive and negative class counts
        P = tf.reduce_sum(y_true) + eps
        N = tf.reduce_sum(1 - y_true) + eps # type: ignore
        # Soft true positives and true negatives
        true_pos = tf.reduce_sum(y_true * y_pred) # type: ignore
        true_neg = tf.reduce_sum((1 - y_true) * (1 - y_pred)) # type: ignore
        # Recalls
        recall_pos = true_pos / P
        recall_neg = true_neg / N
        # Balanced accuracy
        balanced_acc = 0.5 * (recall_pos + recall_neg)
        return 1.0 - balanced_acc

    return loss_fn
