import pytest
import efficient_det.datasets.coco
import efficient_det.datasets.train_data_prep as train_data_prep
import efficient_det.model.anchor as anch
import efficient_det.model
import tensorflow as tf

from efficient_det.geometry.box import Boxes


@pytest.fixture
def coco():
    anchor_size = 4
    anchor_aspects = [
        (1., 1.),
        (.7, 1.4),
        (1.4, 0.7),
    ]
    iou_match_thresh = 0.3
    anchors = anch.build_anchors(anchor_size, num_levels=3, aspects=anchor_aspects)
    prepper = train_data_prep.ImageBasicPreparation(min_scale=0.8, max_scale=1.5, target_shape=512)
    return efficient_det.datasets.coco.Coco(
        anchors=anchors,
        augmentations=None,
        basic_training_prep=prepper,
        iou_thresh=0.5,
        batch_size=6)


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


@pytest.mark.skip
def test_coco_looks_ok(coco, plt):
    # todo fix
    # val doesnt get shuffled
    ds = coco.training_set()
    k = 5
    for j, (image, regressions) in enumerate(ds):
        pass
        break
    plt.suptitle('Examples from training set of coco\ndo the boxes fit?')
    plt.saveas = f"{plt.saveas[:-4]}.png"

