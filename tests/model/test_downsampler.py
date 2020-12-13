import pytest
import tensorflow as tf

from efficient_det.model.components.model_fudgers import Downsampler


def test_downsampler():
    depth = 12
    downsampler = Downsampler(depth=depth, n_extra=3)
    x = [tf.random.uniform([1, 8, 8, 2])]
    y = downsampler(x)
    assert len(y) == 4
    assert y[0].shape == (1, 8, 8, 2)
    assert y[1].shape == (1, 4, 4, depth)
    assert y[2].shape == (1, 2, 2, depth)
    assert y[3].shape == (1, 1, 1, depth)
