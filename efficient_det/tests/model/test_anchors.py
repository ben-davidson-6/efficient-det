import pytest
import tensorflow as tf
import numpy as np

from efficient_det.model.anchor import EfficientDetAnchors, Boxes
from efficient_det.datasets.coco import Coco


def test_row_col_grid():
    grid = EfficientDetAnchors._get_pixel_coords(2, 2)
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
            [4, 12],
            [4, 12.]
        ])
    desired_y = tf.transpose(desired_x)
    desired = tf.stack([desired_x, desired_y], axis=-1)[None, :, :, None]
    yield {'regression': tensor, 'default': desired}


def test_default_absolute_centroids(random_regression):
    anchors = EfficientDetAnchors(size=1, aspects=[(1, 1.)], num_levels=3)

    coords = anchors._default_boxes_for_regression(
        level=0,
        regression=random_regression['regression'])
    tf.debugging.assert_near(random_regression['default'], coords[..., :2])


def test_regressed_centroids(random_regression):
    anchors = EfficientDetAnchors(size=1, aspects=[(1, 1.)], num_levels=3)
    coords = anchors._regress_to_absolute_individual_level(
        level=0,
        regression=random_regression['regression'])
    tf.debugging.assert_none_equal(coords[..., :2], random_regression['default'])

    zero_offset = random_regression['regression']*0
    coords = anchors._regress_to_absolute_individual_level(
        level=0,
        regression=zero_offset)
    tf.debugging.assert_near(coords[..., :2], random_regression['default'])


def test_regressed_shapes():
    scale_x, scale_y = 0.5, 0.1
    regression = tf.stack([0., 0., tf.math.log(scale_x), tf.math.log(scale_y)])
    anchors = EfficientDetAnchors(size=1, aspects=[(1, 1.)], num_levels=3)
    box_shape = anchors._regress_to_absolute_individual_level(
        0,
        regression[None, None, None, None])
    box_shape = box_shape[0, 0, 0, 0, 2:]
    expected_shape = tf.constant([scale_x*8, scale_y*8])/2
    tf.debugging.assert_near(box_shape, expected_shape)


def test_boxes_as_centroids():
    boxes = Boxes(1, 1, tf.constant([[0, 0, 1, 1]], dtype=tf.float32), tf.constant([1]))
    boxes_in_alternate_formed = boxes.boxes_as_centroid_and_widths()
    expected = tf.constant([[0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)
    tf.debugging.assert_near(expected, boxes_in_alternate_formed)


def test_box_area():
    boxes = tf.constant([
        [0, 0, 1, 1],
        [0, 0, 2., 3],
        [3, 3, 6, 6],
        [4, 4, 2, 2]
    ])
    expected_areas = tf.constant([1, 6, 9., 0])
    areas = Boxes.box_area(boxes)
    tf.debugging.assert_near(areas, expected_areas)


def test_box_to_regression_and_back():

    image_height = 100
    image_width = 100
    width = 32
    level = 2
    box = [32, 32, 32 + width, 32 + width]
    expected_regression = tf.constant([0, 0, 0, 0.])
    boxes = Boxes(image_height, image_width, tf.constant([box], dtype=tf.float32), tf.constant([1]))

    anchors = EfficientDetAnchors(size=1, aspects=[(1, 1.)], num_levels=3, iou_match_thresh=0.)

    label, regressions = anchors._assign_boxes_to_level(boxes, level)
    tf.debugging.assert_near(regressions[0, 1, 1, 0], tf.constant(expected_regression))
    box_from_regree = anchors._regress_to_absolute_tlbr(level, regressions)[0, 1, 1, 0]
    tf.debugging.assert_near(tf.cast(box, tf.float32), box_from_regree)


@pytest.mark.skip(reason='visual confirmation')
def test_visual_example():
    import matplotlib.pyplot as plt
    val_ds = Coco.raw_dataset(Coco.validation)
    anchors = EfficientDetAnchors(
        size=4,
        aspects=[(1, 1.), (0.7, 1.4), (1.4, 0.7)],
        num_levels=3,
        iou_match_thresh=0.4)
    for example in val_ds:
        height, width = example['image'].shape[:2]
        gt_boxes = Boxes(height, width, example['objects']['bbox'], example['objects']['label'])
        gt_boxes.unnormalise()

        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        image = tf.image.convert_image_dtype(example['image'][None], tf.float32)
        original_boxes = example['objects']['bbox'][None]
        original_bounding_image = tf.image.draw_bounding_boxes(
            image,
            original_boxes,
            colors,
        )[0]

        absos = anchors.calculate_and_return_absolute_tlbr_boxes(gt_boxes)
        absos /= tf.constant([height, width, height, width], dtype=tf.float32)[None]
        anchor_bbox_image = tf.image.draw_bounding_boxes(
            image,
            absos[None],
            colors,
        )[0]
        plt.imshow(np.vstack((original_bounding_image, anchor_bbox_image)))
        plt.show()
        break

