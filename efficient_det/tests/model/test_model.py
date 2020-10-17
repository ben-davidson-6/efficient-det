import pytest
import tensorflow as tf

from efficient_det.model import EfficientDetNetwork


def test_out_shape(mocker):
    anchors = mocker.Mock()
    n_classes = 3
    n_anchors = 5
    h = w = 256
    anchors.num_boxes = mocker.Mock(return_value=n_anchors)
    net = EfficientDetNetwork(phi=0, num_classes=n_classes, anchors=anchors)
    image = tf.random.uniform(shape=[1, h, w, 3])
    out = net(image)
    assert len(out) == 6
    for i in range(6):
        reduction = (2**(3 + i))
        out_shape = (1, h//reduction, w//reduction, n_anchors, n_classes + 4)
        assert out[i].shape == out_shape


def test_bad_shape(mocker):
    anchors = mocker.Mock()
    anchors.num_boxes = mocker.Mock(return_value=5)
    net = EfficientDetNetwork(phi=0, num_classes=3, anchors=anchors)
    image = tf.random.uniform(shape=[1, 427 - 11, 640, 3])
    out = net(image)[0]
    out_shape = (1, 52, 80, 5, 3 + 4)
    assert out.shape == out_shape
