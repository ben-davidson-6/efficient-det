import pytest
import tensorflow as tf

import efficient_det.model.loss as loss


def test_outputs_correct_shape():
    num_classes = 2
    num_anchors = 1
    l = loss.loss(weights=[0.5, 0.5], alpha=0.25, gamma=1.25, delta=0.1, n_classes=num_classes, prop_neg=2)

    y_true_class = tf.random.uniform((1, 20, 20, num_anchors), minval=-1, maxval=2, dtype=tf.int32)
    y_pred_class = tf.random.uniform((1, 20, 20, num_anchors, num_classes))
    y_true_regression = tf.random.uniform((1, 20, 20, num_anchors, 4))
    y_pred_regression = tf.random.uniform((1, 20, 20, num_anchors, 4))

    out = l((y_true_class, y_true_regression), (y_pred_class, y_pred_regression))
    assert out.ndim == 0


def test_masks_everything_correct_shape():
    num_classes = 2
    num_anchors = 1
    # do not flip any negatives prop_neg=0 and make box loss 0 with weight
    l = loss.loss(weights=[999., 0.], alpha=0.25, gamma=1.25, delta=0.1, n_classes=num_classes, prop_neg=0)
    # all classes are negative
    y_true_class = tf.ones((1, 20, 20, num_anchors), dtype=tf.int32)*-1
    y_pred_class = tf.random.uniform((1, 20, 20, num_anchors, num_classes))
    y_true_regression = tf.random.uniform((1, 20, 20, num_anchors, 4))
    y_pred_regression = tf.random.uniform((1, 20, 20, num_anchors, 4))

    out = l((y_true_class, y_true_regression), (y_pred_class, y_pred_regression))
    tf.debugging.assert_near(out, tf.zeros_like(out))


def test_doesnt_mask_everything():
    num_classes = 2
    num_anchors = 1
    # do not flip any negatives prop_neg=0 and make box loss 0 with weight
    l = loss.loss(weights=[.5, 0.5], alpha=0.25, gamma=1.25, delta=0.1, n_classes=num_classes, prop_neg=1)
    # all classes are negative
    y_true_class = tf.random.uniform((1, 20, 20, num_anchors), minval=-1, maxval=2, dtype=tf.int32)
    y_pred_class = tf.random.uniform((1, 20, 20, num_anchors, num_classes))
    y_true_regression = tf.random.uniform((1, 20, 20, num_anchors, 4))
    y_pred_regression = tf.random.uniform((1, 20, 20, num_anchors, 4))

    out = l((y_true_class, y_true_regression), (y_pred_class, y_pred_regression))
    assert out > 0

