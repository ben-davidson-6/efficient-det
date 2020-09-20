import pytest
import tensorflow as tf

import efficient_det.model.loss as loss


def test_outputs_correct_shape():
    num_classes = 2
    num_anchors = 1
    l = loss.loss(weights=[0.5, 0.5], alpha=0.25, gamma=1.25, delta=0.1, n_classes=num_classes, prop_neg=2)
    y_true = tf.random.uniform((1, 20, 20, num_anchors, 1 + 4), minval=-1, maxval=2, dtype=tf.int32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.random.uniform((1, 20, 20, num_anchors, num_classes + 4), minval=-1, maxval=2, dtype=tf.float32)
    out = l(y_true, y_pred)
    assert out.ndim == 0


def test_masks_everything_correct_shape():
    num_classes = 2
    num_anchors = 1
    # do not flip any negatives prop_neg=0 and make box loss 0 with weight
    l = loss.loss(weights=[999., 0.], alpha=0.25, gamma=1.25, delta=0.1, n_classes=num_classes, prop_neg=0)

    y_true = tf.ones((1, 20, 20, num_anchors, 5))*-1
    y_pred = tf.random.uniform((1, 20, 20, num_anchors, num_classes + 4), minval=-1, maxval=2, dtype=tf.float32)

    out = l(y_true, y_pred)
    tf.debugging.assert_near(out, tf.zeros_like(out))


def test_doesnt_mask_everything():
    num_classes = 2
    num_anchors = 1
    # do not flip any negatives prop_neg=0 and make box loss 0 with weight
    l = loss.loss(weights=[.5, 0.5], alpha=0.25, gamma=1.25, delta=0.1, n_classes=num_classes, prop_neg=1)
    # all classes are negative
    y_true = tf.random.uniform((1, 20, 20, num_anchors, 1 + 4), minval=-1, maxval=2, dtype=tf.int32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.random.uniform((1, 20, 20, num_anchors, num_classes + 4), minval=-1, maxval=2, dtype=tf.float32)

    out = l(y_true, y_pred)
    assert out > 0

