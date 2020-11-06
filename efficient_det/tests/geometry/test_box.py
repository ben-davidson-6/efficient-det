import tensorflow as tf
import pytest

from efficient_det.geometry.box import TLBRBoxes, DefaultAnchorBoxes, CentroidWidthBoxes, DefaultAnchorOffsets
from efficient_det.geometry.common import pixel_coordinates


@pytest.fixture()
def anchor_box():
    params = dict(
        stride_height=8,
        stride_width=4,
        grid_height=4,
        grid_width=6,
        box_height=2,
        box_width=4)
    default_anchor = DefaultAnchorBoxes(**params)
    return default_anchor, params


def test_row_col_grid():
    grid = pixel_coordinates(2, 2)
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


def test_default_absolute_centroids(anchor_box):
    default_anchor, params = anchor_box
    tf.debugging.assert_near(default_anchor.tensor[0, 2:], tf.constant([params['box_width'] / 2, params['box_height'] / 2]))
    assert default_anchor.as_original_shape().shape == tf.TensorShape((params['grid_height'], params['grid_width'], 4))


def test_go_between_tlbr_and_centroid():
    actual_tlbr_box = tf.constant([0., 0., 1., 1.])
    actual_cw_box = tf.constant([0.5, 0.5, 0.5, 0.5])

    # from tlbr
    tlbr_box = TLBRBoxes(actual_tlbr_box)
    tlbr_cw_box = tlbr_box.as_centroid_and_width_box()
    tlbr_cw_tlbr_box = tlbr_cw_box.as_tlbr_box()
    tf.debugging.assert_near(tlbr_box.tensor, tlbr_cw_tlbr_box.tensor)
    tf.debugging.assert_near(actual_cw_box[None], tlbr_cw_box.tensor)

    # from cw
    cw_box = CentroidWidthBoxes(actual_cw_box)
    cw_tlbr_box = cw_box.as_tlbr_box()
    cw_tlbr_cw_box = cw_tlbr_box.as_centroid_and_width_box()
    tf.debugging.assert_near(cw_box.tensor, cw_tlbr_cw_box.tensor)
    tf.debugging.assert_near(actual_tlbr_box[None], cw_tlbr_box.tensor)


def test_iou():
    # box full overlap
    unit_box_tensor = tf.constant([0., 0., 1., 1.])
    unit_tlbr = TLBRBoxes(unit_box_tensor)
    iou = unit_tlbr.iou(unit_tlbr)
    tf.debugging.assert_near(iou, 1.)

    # box no overlap
    a_tensor = tf.constant([0., 1., 1., 2.])
    a = TLBRBoxes(a_tensor)
    iou = unit_tlbr.iou(a)
    tf.debugging.assert_near(iou, 0.)

    # box some overlap
    b_tensor = tf.constant([0., 0., 1., 3.])
    b = TLBRBoxes(b_tensor)
    iou = unit_tlbr.iou(b)
    tf.debugging.assert_near(iou, 1 / 3.)

    # check works for all
    tlbr = TLBRBoxes(tf.stack([unit_box_tensor, a_tensor, b_tensor]))
    ious = unit_tlbr.iou(tlbr)
    tf.debugging.assert_near(ious, tf.constant([1., 0., 1 / 3.]))


def test_anchor_offsets(anchor_box):
    default_anchor, params = anchor_box
    offsets = default_anchor.as_offset_boxes(default_anchor)
    anchor_offsets = DefaultAnchorOffsets(offsets)
    default_anchor_from_offset = anchor_offsets.as_centroid_width_box(default_anchor)

    offset_tensor = tf.zeros([params['grid_height']*params['grid_width'], 4])
    tf.debugging.assert_near(offsets, offset_tensor)
    tf.debugging.assert_near(default_anchor_from_offset.tensor, default_anchor.tensor)


def test_box_to_regression_and_back(anchor_box):
    unit_box_tensor = tf.constant([0., 0., 1., 1.])
    unit_tlbr = TLBRBoxes(unit_box_tensor)
    default_anchor, params = anchor_box
    offset_tensor = unit_tlbr.as_centroid_and_width_box().as_offset_boxes(default_anchor)
    offset = DefaultAnchorOffsets(offset_tensor)
    back_to_tlbr = offset.as_centroid_width_box(default_anchor).as_tlbr_box()
    tf.debugging.assert_near(back_to_tlbr.tensor, unit_tlbr.tensor)
