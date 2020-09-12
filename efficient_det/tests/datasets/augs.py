import pytest
import tensorflow as tf
import efficient_det.datasets.augs as augs
import efficient_det.model.anchor as anchor
import random
import matplotlib.pyplot as plt


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
    prepper = augs.ImageBasicPreparation(min_scale=1, max_scale=1, target_shape=64)
    image, bboxes, labels = prepper.scale_and_random_crop(*single_example)
    assert labels.shape == single_example[2].shape
    assert bboxes.shape == single_example[1].shape
    assert image.shape == single_example[0].shape


def test_scale_image(single_example):
    image, bbox, labels = single_example
    scale = 0.7
    prepper = augs.ImageBasicPreparation(min_scale=scale, max_scale=scale, target_shape=64)
    image_mod, bboxes_mod = prepper._random_scale_image(image, bbox)
    tf.debugging.assert_equal(bboxes_mod[0], tf.cast(tf.constant([0., 0., scale*64, scale*64]), tf.int32))


@pytest.fixture
def random_example_in_image(actual_image):
    # h = random.randint(64, 128)
    # w = random.randint(64, 128)
    # image = tf.image.resize(actual_image, size=tf.stack([h, w]))
    h, w = actual_image.shape[:2]
    bboxes = []
    colors = []
    n_boxes = 10

    for _ in range(n_boxes):
        ymin = random.randint(0, h)
        xmin = random.randint(0, w)
        ymax = random.randint(ymin, h)
        xmax = random.randint(xmin, w)
        bboxes.append([ymin, xmin, ymax, xmax])
        colors.append([random.random(), random.random(), random.random()])

    bboxes = tf.stack(bboxes, axis=0)
    colors = tf.stack(colors, axis=0)
    labels = tf.ones([n_boxes, 1], dtype=tf.int32)

    return actual_image, bboxes, labels, colors


@pytest.fixture
def box_plotter():
    def plotter(image, image_mod, bboxes, bboxes_mod, colors, title):
        bboxes = anchor.Boxes(image.shape[0], image.shape[1], bboxes, None)
        bboxes_mod = anchor.Boxes(image_mod.shape[0], image_mod.shape[1], bboxes_mod, None)
        bboxes.normalise()
        bboxes_mod.normalise()

        original = tf.image.draw_bounding_boxes(image[None], bboxes.box_tensor[None], colors)[0]
        cropped_etc = tf.image.draw_bounding_boxes(image_mod[None], bboxes_mod.box_tensor[None], colors)[0]
        plt.suptitle(title)
        plt.subplot(2, 1, 1)
        plt.imshow(original)
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(cropped_etc)
        plt.axis('off')
        plt.show()
    return plotter


@pytest.mark.skip(reason='visual test')
def test_looks_right(random_example_in_image, box_plotter):
    image, bboxes, labels, colors = random_example_in_image

    for _ in range(5):
        prepper = augs.ImageBasicPreparation(min_scale=0.5, max_scale=2., target_shape=512)
        image_mod, bboxes_mod, labels_mod = prepper.scale_and_random_crop(image, bboxes, labels)
        box_plotter(image, image_mod, bboxes, bboxes_mod, colors, title='random example')


@pytest.mark.skip('visual check')
def test_scaling_looks_right(random_example_in_image, box_plotter):
    image, bboxes, labels, colors = random_example_in_image

    # no scale
    prepper = augs.ImageBasicPreparation(min_scale=1, max_scale=1, target_shape=64)
    image_mod, bboxes_mod = prepper._random_scale_image(image, bboxes)
    box_plotter(image, image_mod, bboxes, bboxes_mod, colors, 'no scaling')

    # up scale
    prepper = augs.ImageBasicPreparation(min_scale=2., max_scale=2., target_shape=64)
    image_mod, bboxes_mod = prepper._random_scale_image(image, bboxes)
    box_plotter(image, image_mod, bboxes, bboxes_mod, colors, 'up scaling')

    # down scale
    prepper = augs.ImageBasicPreparation(min_scale=0.5, max_scale=0.5, target_shape=64)
    image_mod, bboxes_mod = prepper._random_scale_image(image, bboxes)
    box_plotter(image, image_mod, bboxes, bboxes_mod, colors, 'down scaling')

@pytest.mark.skip(reason='visual test')
def test_pad_to_looks_right(random_example_in_image, box_plotter):
    image, bboxes, labels, colors = random_example_in_image
    h, w = image.shape[:2]
    min_dim = min(h, w)
    max_dim = max(h, w)

    # single dim
    prepper = augs.ImageBasicPreparation(min_scale=1, max_scale=1, target_shape=min_dim + 100)
    image_mod, bboxes_mod = prepper._pad_to_target_if_needed(image, bboxes)
    box_plotter(image, image_mod, bboxes, bboxes_mod, colors, 'with padding single dim')

    # both dim
    prepper = augs.ImageBasicPreparation(min_scale=1, max_scale=1, target_shape=max_dim + 100)
    image_mod, bboxes_mod = prepper._pad_to_target_if_needed(image, bboxes)
    box_plotter(image, image_mod, bboxes, bboxes_mod, colors, 'with padding both dim')

    # 0 padding
    prepper = augs.ImageBasicPreparation(min_scale=2., max_scale=2., target_shape=min_dim)
    image_mod, bboxes_mod = prepper._pad_to_target_if_needed(image, bboxes)
    box_plotter(image, image_mod, bboxes, bboxes_mod, colors, 'zero padding')

    # ngative
    prepper = augs.ImageBasicPreparation(min_scale=0.5, max_scale=0.5, target_shape=64)
    image_mod, bboxes_mod = prepper._pad_to_target_if_needed(image, bboxes)
    box_plotter(image, image_mod, bboxes, bboxes_mod, colors, 'negative?')


@pytest.mark.skip(reason='visual test')
def test_crop_looks_right(random_example_in_image, box_plotter, mocker):
    image, bboxes, labels, colors = random_example_in_image
    h, w = image.shape[:2]
    min_dim = min(h, w)
    max_dim = max(h, w)

    # single dim
    prepper = augs.ImageBasicPreparation(min_scale=1, max_scale=1, target_shape=min_dim)
    image_mod, bboxes_mod, labels = prepper._random_crop(image, bboxes, labels)
    box_plotter(image, image_mod, bboxes, bboxes_mod, colors, 'with padding single dim')


