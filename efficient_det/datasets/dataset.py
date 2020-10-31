import tensorflow as tf
import efficient_det

from efficient_det.model.anchor import Anchors
from efficient_det.datasets.augs import Augmenter
from efficient_det.datasets.train_data_prep import ImageBasicPreparation
from efficient_det.geometry.box import TLBRBoxes
from efficient_det.geometry.common import level_to_stride


class Dataset:
    validation = 'validation'
    train = 'train'

    def __init__(
            self,
            anchors: Anchors,
            augmentations: Augmenter,
            basic_training_prep: ImageBasicPreparation,
            iou_thresh: float,
            batch_size: int):
        self.basic_training_prep = basic_training_prep
        self.batch_size = batch_size
        self.anchors = anchors
        self.iou_thresh = iou_thresh
        self.augmentations = lambda x, y: (x, y) if augmentations is None else augmentations

    def training_set(self):
        ds = self._raw_training_set()
        ds = ds.map(self.training_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        # ds = ds.map(self.augmentations, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

    @tf.function
    def training_transform(self, *args):
        return self._build_offset_boxes(*self.basic_training_prep.scale_and_random_crop(*args))

    def validation_set(self):
        ds = self._raw_validation_set()
        ds = ds.map(self.validation_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(1).prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    @tf.function
    def validation_transform(self, *args):
        return self._build_offset_boxes(*self._closest_acceptable_multiple(*args))

    def _build_offset_boxes(self, image, bboxes, labels):
        """offset is  a single tensor with final dim = 5 = class + regression"""
        height, width, channel = image.shape
        bboxes = TLBRBoxes(bboxes)
        if bboxes.is_empty():
            offset = self._empty_offset(height, width)
        else:
            offsets_ious_boxes = self.anchors.to_offset_tensors(bboxes, height, width)
            offsets, ious, matched_boxes = zip(*offsets_ious_boxes)
            labels = [tf.gather(tf.cast(labels, tf.float32), box) for box in matched_boxes]
            ious = [x > self.iou_thresh for x in ious]
            # note here we add 1 so that the background class is zero!
            labels = [tf.where(iou, label, efficient_det.NO_CLASS_LABEL) + 1 for iou, label in zip(ious, labels)]
            offset = tuple([tf.concat([label[..., None], offset], axis=-1) for label, offset in zip(labels, offsets)])
        return image, offset

    def _closest_acceptable_multiple(self, image, box, label):
        # todo hack for the time being
        image = tf.image.resize(image, (512, 512))
        return image, box, label

    def _empty_offset(self, height, width):
        offset = []
        for i in range(self.anchors.num_levels):
            stride = level_to_stride(i)
            empty_level = tf.ones([height // stride, width // stride, len(self.anchors.aspects), 5]) * efficient_det.NO_CLASS_LABEL
            offset.append(empty_level)
        return tuple(offset)

    def _raw_validation_set(self,):
        raise NotImplemented

    def _raw_training_set(self,):
        raise NotImplemented

