import pytest
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from efficient_det.datasets.coco import Coco
from efficient_det.model.anchor import build_anchors
from efficient_det.geometry.box import TLBRBoxes


def test_from_raw_to_regression_and_back_visual(plt):
    val_ds = Coco.raw_dataset(Coco.validation)
    anchors = build_anchors(4, num_levels=3, aspects=[(1., 1.)])
    for k, example in enumerate(val_ds):
        image = example['image']
        height, width = image.numpy().shape[:2]

        original = tf.image.draw_bounding_boxes(
            tf.image.convert_image_dtype(image[None], tf.float32),
            example['objects']['bbox'][None],
            tf.constant([(0., 0., 1.)]))[0]

        bbox = TLBRBoxes(example['objects']['bbox'])
        offsets = anchors.to_offset_tensors(bbox, height, width)
        offsets, ious, matched_boxes = zip(*offsets)

        # we only turn them back when they have been from the network so they should  have a  batach
        offsets = [x[None] for x in offsets]
        new_boxes = tf.concat([tf.reshape(x, [1, -1, 4]) for x in anchors.to_tlbr_tensor(offsets)], axis=1)
        new = tf.image.draw_bounding_boxes(
            tf.image.convert_image_dtype(image, tf.float32)[None],
            new_boxes,
            tf.constant([(0., 0., 1.)]))[0]

        combined = tf.concat([original, new], axis=0)
        plt.imshow(combined)
        plt.show()
        break
    plt.saveas = f"{plt.saveas[:-4]}.png"



