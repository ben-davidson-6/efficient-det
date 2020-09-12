import tensorflow as tf

from efficient_det.model.anchor import Boxes, EfficientDetAnchors
from efficient_det.datasets.augs import Augmenter
from efficient_det.datasets.train_data_prep import ImageBasicPreparation


class Dataset:
    validation = 'validation'
    train = 'train'
    # todo tune the maps
    def __init__(
            self,
            anchors: EfficientDetAnchors,
            augmentations: Augmenter,
            basic_training_prep: ImageBasicPreparation,
            batch_size: int):
        self.augmentations = augmentations
        self.basic_training_prep = basic_training_prep
        self.batch_size = batch_size
        self.anchors = anchors

    def training_set(self):
        ds = Dataset._raw_validation_set()
        ds = ds.map(self.basic_training_prep)
        ds = ds.batch(self.batch_size)
        ds = ds.batch(self.augmentations)
        ds = ds.map(self._build_regressions).prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def validation_set(self):
        ds = Dataset._raw_validation_set()
        ds = ds.map(self._build_regressions)
        ds = ds.batch(1).prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def _build_regressions(self, image, bboxes, labels):
        bboxes = Boxes.from_image_boxes_labels(image, bboxes, labels)
        labels, regressions = self.anchors.absolute_to_regression(bboxes)
        regressions = regressions[0]
        return image, regressions, labels

    @staticmethod
    def _raw_validation_set():
        raise NotImplemented

    @staticmethod
    def _raw_training_set():
        raise NotImplemented
