import tensorflow as tf

from efficient_det.model.anchor import EfficientDetAnchors
from efficient_det.common.box import Boxes
from efficient_det.datasets.augs import Augmenter
from efficient_det.datasets.train_data_prep import ImageBasicPreparation


class Dataset:
    validation = 'validation'
    train = 'train'

    def __init__(
            self,
            anchors: EfficientDetAnchors,
            augmentations: Augmenter,
            basic_training_prep: ImageBasicPreparation,
            batch_size: int):
        self.basic_training_prep = basic_training_prep
        self.batch_size = batch_size
        self.anchors = anchors
        self.augmentations = lambda x, y: (x, y) if augmentations is None else augmentations

    def training_set(self):
        ds = self._raw_training_set()
        ds = ds.map(self.basic_training_prep.scale_and_random_crop_normalised)
        ds = ds.map(self._build_regressions)
        ds = ds.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.map(self.augmentations)
        return ds

    def validation_set(self):
        ds = self._raw_validation_set()
        ds = ds.map(self._build_regressions)
        ds = ds.batch(1).prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def _build_regressions(self, image, bboxes, labels):
        bboxes = Boxes.from_image_boxes_labels(image, bboxes, labels)
        regressions = self.anchors.absolute_to_regression(bboxes)
        return image, regressions

    def _raw_validation_set(self,):
        raise NotImplemented

    def _raw_training_set(self,):
        raise NotImplemented
