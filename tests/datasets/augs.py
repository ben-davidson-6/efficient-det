import pytest
import tensorflow as tf
import efficient_det.datasets.augs as augs
import efficient_det.datasets.train_data_prep
import random

from efficient_det.geometry.plot import draw_image_with_boxes


@pytest.fixture(scope='module')
def single_example():
    h = w = 64
    image = tf.random.uniform([h, w, 3])

    # tlbr
    bbox = tf.constant([
        [0., 0., 1., 1.],  # whole image
        [0.1, 0.1, 0.2, 0.2],  # small square in upper left
        [0.3, 0.1, 0.9, 0.9]  # bigger rectangle
    ])
    labels = tf.constant([0, 1, 2], dtype=tf.int32)
    return image, bbox, labels


def test_does_nothing(single_example):
    prepper = efficient_det.datasets.train_data_prep.ImageBasicPreparation(overlap_percentage=0.3, min_scale=1., max_scale=1., target_shape=64)
    image, bboxes, labels = prepper.scale_and_random_crop(*single_example)
    assert labels.shape == single_example[2].shape
    assert bboxes.shape == single_example[1].shape
    assert image.shape == single_example[0].shape


@pytest.fixture
def random_example_in_image(actual_image):
    h, w = actual_image.shape[:2]
    bboxes = []
    n_boxes = 10

    for _ in range(n_boxes):
        ymin = random.random()
        xmin = random.random()
        ymax = random.uniform(ymin, 1.)
        xmax = random.uniform(xmin, 1.)
        bboxes.append([ymin, xmin, ymax, xmax])

    bboxes = tf.stack(bboxes, axis=0)
    labels = tf.ones([n_boxes, 1], dtype=tf.int32)

    return actual_image, bboxes, labels


def test_augmenting_looks_good(random_example_in_image, plt):
    image, bboxes, labels = random_example_in_image
    plt.subplot(2, 1, 1)
    plt.imshow(draw_image_with_boxes(image, bboxes))
    for k in range(3):
        prepper = efficient_det.datasets.train_data_prep.ImageBasicPreparation(overlap_percentage=0.3, min_scale=0.5, max_scale=2., target_shape=512)
        image_mod, bboxes_mod, labels_mod = prepper.scale_and_random_crop(image, bboxes, labels)
        plt.subplot(2, 3, 4 + k)
        plt.imshow(draw_image_with_boxes(image_mod, bboxes_mod))
    plt.suptitle('Cropping and scaling while keeping the boxes\n should make sense')
    plt.saveas = f"{plt.saveas[:-4]}.png"


