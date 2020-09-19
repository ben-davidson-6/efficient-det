import pytest
import tensorflow as tf

from efficient_det.model import EfficientDetNetwork


def test_out_shape(mocker):
    anchors = mocker.Mock()
    anchors.num_boxes = mocker.Mock(return_value=5)
    net = EfficientDetNetwork(phi=0, num_classes=3, anchors=anchors)
    image = tf.random.uniform(shape=[1, 256, 256, 3])
    out = net(image)
    start_label_shape = (1, 32, 32, 5, 3)
    start_regre_shape = (1, 32, 32, 5, 4)
    label, regression = out[0]
    assert label.shape == start_label_shape
    assert regression.shape == start_regre_shape

