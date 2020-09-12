import pytest
import tensorflow as tf

import efficient_det.model.loss as loss


def test_outputs_correct_shape():
    num_classes = 2
    num_anchors = 1
    l = loss.loss(weights=[0.5, 0.5], alpha=0.25, gamma=1.25)

    y_true_class = tf.random.uniform((1, 20, 20, num_classes*num_anchors))
    y_pred_class = tf.random.uniform((1, 20, 20, num_classes*num_anchors))
    y_true_regression = tf.random.uniform((1, 20, 20, num_anchors*4))
    y_pred_regression = tf.random.uniform((1, 20, 20, num_anchors*4))

    out = l((y_true_class, y_true_regression), (y_pred_class, y_pred_regression))
    assert out.ndim == 0




