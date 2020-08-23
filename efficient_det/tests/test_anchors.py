import pytest
import tensorflow as tf
import numpy as np

from efficient_det.model.anchor import EfficientDetAnchors


def test_row_col_grid():
    tensor = tf.random.uniform([1, 2, 2, 1, 1])
    grid = EfficientDetAnchors._get_pixel_coords(tensor)
    x = grid[..., 0]
    x_desired = tf.constant([
        [0, 1.],
        [0, 1]
    ])
    y = grid[..., 1]
    y_desired = tf.transpose(x_desired)

    assert grid.shape[-1] == 2
    assert grid.ndim == 3
    assert tf.reduce_all(x == x_desired).numpy()
    assert tf.reduce_all(y == y_desired).numpy()


@pytest.fixture
def random_regression():
    # batch, height, width, n, box
    tensor = tf.random.uniform([1, 2, 2, 1, 4])
    desired_x = tf.constant([
            [0, 8],
            [0, 8.]
        ])
    desired_y = tf.transpose(desired_x)
    desired = tf.stack([desired_x, desired_y], axis=-1)[None, :, :, None]
    yield {'regression': tensor, 'default': desired}


def test_default_absolute_centroids(random_regression):
    coords = EfficientDetAnchors._default_absolute_centroids(
        level=0,
        tensor=random_regression['regression'])
    tf.debugging.assert_near(random_regression['default'], coords)


def test_regressed_centroids(random_regression):
    coords = EfficientDetAnchors._regressed_centroids(
        level=0,
        regression=random_regression['regression'])
    tf.debugging.assert_none_equal(coords, random_regression['default'])

    zero_offset = random_regression['regression']*0
    coords = EfficientDetAnchors._regressed_centroids(
        level=0,
        regression=zero_offset)
    tf.debugging.assert_near(coords, random_regression['default'])


def test_regressed_shapes():
    scale_x, scale_y = 0.5, 0.1
    regression = tf.stack([0., 0., tf.math.log(scale_x), tf.math.log(scale_y)])
    anchors = EfficientDetAnchors(size=1, aspects=[(1, 1.)])
    box_shape = anchors._regressed_box_shapes(
        0,
        regression[None, None, None, None])
    box_shape = box_shape[0, 0, 0, 0]
    expected_shape = tf.constant([scale_x*8, scale_y*8])/2
    tf.debugging.assert_near(box_shape, expected_shape)


def test_from_regression_to_absolute():
    scale_x, scale_y = 0.5, 0.1
    regression = tf.stack([0., 0., tf.math.log(scale_x), tf.math.log(scale_y)])
    anchors = EfficientDetAnchors(size=1, aspects=[(1, 1.)])
    box_shape = anchors._regressed_box_shapes(
        0,
        regression[None, None, None, None])
    box_shape = box_shape[0, 0, 0, 0]
    expected_shape = tf.constant([scale_x*8, scale_y*8])/2
    tf.debugging.assert_near(box_shape, expected_shape)
