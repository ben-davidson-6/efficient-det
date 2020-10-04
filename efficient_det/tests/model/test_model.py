import pytest
import tensorflow as tf

from efficient_det.model import EfficientDetNetwork


def test_out_shape(mocker):
    anchors = mocker.Mock()
    anchors.num_boxes = mocker.Mock(return_value=5)
    net = EfficientDetNetwork(phi=0, num_classes=3, anchors=anchors)
    image = tf.random.uniform(shape=[1, 256, 256, 3])
    out = net(image)[0]
    out_shape = (1, 32, 32, 5, 3 + 4)
    assert out.shape == out_shape


def test_bad_shape(mocker):
    anchors = mocker.Mock()
    anchors.num_boxes = mocker.Mock(return_value=5)
    net = EfficientDetNetwork(phi=0, num_classes=3, anchors=anchors)
    image = tf.random.uniform(shape=[1, 427 - 11, 640, 3])
    out = net(image)[0]
    out_shape = (1, 52, 80, 5, 3 + 4)
    assert out.shape == out_shape
