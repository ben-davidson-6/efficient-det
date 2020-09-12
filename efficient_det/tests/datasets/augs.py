import pytest
import tensorflow as tf
import efficient_det.datasets.augs as augs


@pytest.fixture(scope='module')
def image_bbox_and_desired_crop():
    h = w = 64
    image = tf.random.uniform([h, w, 3])

    # tlbr
    bbox = tf.constant([
        [0., 0., 1., 1.],  # whole image
        [0.1, 0.1, 0.2, 0.2],  # small square in upper left
        [0.3, 0.1, 0.9, 0.9]  # bigger rectangle
    ])

    # to crop
    crop = tf.constant([0.2, 0.2, 0.8, 0.8])
    return image


def test_crop(image_and_bbox):
    image, bbox = augs.random_crop_image_and_bbox(*image_and_bbox)
