import pytest
import tensorflow as tf

from efficient_det.model import EfficientDetNetwork


def test_out_shape():
    net = EfficientDetNetwork(phi=0, num_classes=3, num_anchors=5)
    image = tf.random.uniform(shape=[1, 256, 256, 3])
    out = net(image)[0]
    out_shape = (1, 32, 32, 5, 3 + 4)
    assert out.shape == out_shape

