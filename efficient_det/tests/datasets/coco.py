import pytest
import efficient_det.datasets.coco
import efficient_det.datasets.train_data_prep as train_data_prep
import efficient_det.model
import tensorflow as tf
import matplotlib.pyplot as plt

from efficient_det.common.plot import Plotter
from efficient_det.common.box import Boxes


@pytest.fixture
def coco():
    anchor_size = 4
    anchor_aspects = [
        (1., 1.),
        (.7, 1.4),
        (1.4, 0.7),
    ]
    iou_match_thresh = 0.
    anchors = efficient_det.model.EfficientDetAnchors(
        anchor_size,
        anchor_aspects,
        num_levels=3,
        iou_match_thresh=iou_match_thresh)
    prepper = train_data_prep.ImageBasicPreparation(min_scale=0.8, max_scale=1.5, target_shape=512)
    return efficient_det.datasets.coco.Coco(
        anchors=anchors,
        augmentations=None,
        basic_training_prep=prepper,
        batch_size=2)


def test_types_and_shapes(coco):
    for image, bboxes, labels in coco._raw_training_set():
        assert image.ndim == 3
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert labels.shape[0] == bboxes.shape[0]
        assert image.dtype == tf.uint8, 'image type'
        assert bboxes.dtype == tf.float32, 'box type'
        assert labels.dtype == tf.int64, 'label type'
        break

    for image, bboxes, labels in coco._raw_validation_set():
        assert image.ndim == 3
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert labels.shape[0] == bboxes.shape[0]
        assert image.dtype == tf.uint8, 'image type'
        assert bboxes.dtype == tf.float32, 'box type'
        assert labels.dtype == tf.int64, 'label type'
        break


def test_looks_ok(coco):
    # val doesnt get shuffled
    ds = coco.training_set()
    k = 5
    for j, (image, regressions) in enumerate(ds):
        first_image = image[0]
        first_regression = [(x[0], y[0]) for x, y in regressions]
        absos, labels = coco.anchors.regressions_to_tlbr(first_regression)
        boxes = Boxes.from_image_and_boxes(first_image, absos)
        plotter = Plotter(first_image/255, boxes)
        plotter.plot()
        plt.show()
        if j == k:
            break

