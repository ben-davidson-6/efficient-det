import tensorflow_datasets as tfds
import efficient_det.model.anchor as efficient_det_anchors
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

NAME = 'coco/2017'


def validation_dataset(batch_size):
    return tfds.load(
        NAME,
        download=False,
        batch_size=batch_size,
        split='validation')


anchors = efficient_det_anchors.EfficientDetAnchors(
    size=4,
    aspects=[(1, 1.), (0.7, 1.4), (1.4, 0.7)],
    num_levels=3,
    iou_match_thresh=0.4)
val_ds = tfds.load(NAME,
        download=False,
        split='validation')

for example in val_ds:
    height, width = example['image'].shape[:2]
    gt_boxes = efficient_det_anchors.Boxes(height, width, example['objects']['bbox'], example['objects']['label'])
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
