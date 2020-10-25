import pytest
import tensorflow as tf
import efficient_det.datasets.augs as augs
import efficient_det.datasets.train_data_prep
import random

from efficient_det.geometry.box import Boxes
from efficient_det.geometry.plot import Plotter


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
    bbox *= h
    bbox = tf.cast(bbox, tf.int32)
    labels = tf.constant([0, 1, 2], dtype=tf.int32)
    return image, bbox, labels


def test_does_nothing(single_example):
    prepper = efficient_det.datasets.train_data_prep.ImageBasicPreparation(min_scale=1, max_scale=1, target_shape=64)
    image, bboxes, labels = prepper.scale_and_random_crop(*single_example)
    assert labels.shape == single_example[2].shape
    assert bboxes.shape == single_example[1].shape
    assert image.shape == single_example[0].shape


def test_scale_image(single_example):
    image, bbox, labels = single_example
    scale = 0.7
    prepper = efficient_det.datasets.train_data_prep.ImageBasicPreparation(min_scale=scale, max_scale=scale, target_shape=64)
    image_mod, bboxes_mod = prepper._random_scale_image(image, bbox)
    tf.debugging.assert_equal(bboxes_mod[0], tf.cast(tf.constant([0., 0., scale*64, scale*64]), tf.int32))


@pytest.fixture
def random_example_in_image(actual_image):
    h, w = actual_image.shape[:2]
    bboxes = []
    n_boxes = 10

    for _ in range(n_boxes):
        ymin = random.randint(0, h)
        xmin = random.randint(0, w)
        ymax = random.randint(ymin, h)
        xmax = random.randint(xmin, w)
        bboxes.append([ymin, xmin, ymax, xmax])

    bboxes = tf.stack(bboxes, axis=0)
    labels = tf.ones([n_boxes, 1], dtype=tf.int32)

    return actual_image, bboxes, labels


def test_augmenting_looks_good(random_example_in_image, plt):
    image, bboxes, labels = random_example_in_image
    plot_original = Plotter(image, Boxes.from_image_and_boxes(image, bboxes))
    plot_original.plot((3, 2, 1), 'ORIGINAL', plt)

    for k in range(5):
        prepper = efficient_det.datasets.train_data_prep.ImageBasicPreparation(min_scale=0.5, max_scale=2., target_shape=512)
        image_mod, bboxes_mod, labels_mod = prepper.scale_and_random_crop(image, bboxes, labels)
        bboxes_mod = Boxes.from_image_and_boxes(image_mod, bboxes_mod)
        plot_mod = Plotter(image_mod, bboxes_mod)
        plot_mod.plot((3, 2, k + 2), f'example {k}', plt)
    plt.suptitle('Cropping and scaling while keeping the boxes\n should make sense')
    plt.saveas = f"{plt.saveas[:-4]}.png"


