import tensorflow as tf
import efficient_det

from efficient_det.model.anchor import Anchors
from efficient_det.datasets.augs import Augmenter
from efficient_det.datasets.train_data_prep import ImageBasicPreparation
from efficient_det.geometry.box import TLBRBoxes
from efficient_det.geometry.common import level_to_stride
from efficient_det.constants import NUM_LEVELS


class Dataset:
    validation = 'validation'
    train = 'train'

    def __init__(
            self,
            anchors: Anchors,
            augmentations: Augmenter,
            image_size: int,
            pos_iou_thresh: float,
            neg_iou_thresh: float,
            batch_size: int):
        self.image_size = image_size
        self.batch_size = batch_size
        self.anchors = anchors
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.augmentations = augmentations

    def training_set(self):
        ds = self._raw_training_set()
        ds = ds.map(self.augmentations, num_parallel_calls=2)
        ds = ds.map(self.training_transform, num_parallel_calls=2)
        ds = ds.batch(self.batch_size).prefetch(1)
        return ds

    def training_transform(self, *args):
        return self._build_offset_boxes(*self.resize(*args))

    def validation_set(self):
        ds = self._raw_validation_set()
        ds = ds.map(self.training_transform, num_parallel_calls=2)
        ds = ds.batch(self.batch_size).prefetch(1)
        return ds

    def validation_transform(self, *args):
        return self._build_offset_boxes(*self.resize(*args))

    def resize(self, image, bbox, labels):
        image = tf.image.resize(image, (self.image_size, self.image_size), preserve_aspect_ratio=True)
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        to_pad_left = (self.image_size - width)//2
        to_pad_top = (self.image_size - height)//2
        offset = tf.cast(tf.stack([to_pad_top, to_pad_left]), tf.float32)
        bbox = TLBRBoxes(bbox)
        bbox.unnormalise(height, width)
        bbox.add_offset(offset)
        bbox.normalise(self.image_size, self.image_size)
        image = tf.image.resize_with_crop_or_pad(image, self.image_size, self.image_size)
        return image, bbox.tensor, labels

    def _build_offset_boxes(self, image, bboxes, labels):
        """offset is  a single tensor with final dim = 5 = class + regression"""
        height, width, channel = image.shape
        bboxes = TLBRBoxes(bboxes)
        if bboxes.is_empty():
            offset = self._empty_offset(height, width)
        else:
            offset = self._non_empty_offset(bboxes, labels, height, width)
        return image, offset

    def _non_empty_offset(self, bboxes, labels, height, width):
        offsets_ious_boxes = self.anchors.to_offset_tensors(bboxes, height, width)
        offsets, ious, matched_boxes = zip(*offsets_ious_boxes)
        labels = [tf.gather(tf.cast(labels, tf.float32), box) for box in matched_boxes]
        positive_ious = [x > self.pos_iou_thresh for x in ious]
        n_boxes = tf.shape(bboxes.get_tensor())[0]
        forced_matches = self._forced_matches(n_boxes, matched_boxes, ious)

        positive_ious = [tf.logical_or(p, forced_match) for p, forced_match in zip(positive_ious, forced_matches)]
        negative_ious = [tf.logical_and(x <= self.neg_iou_thresh, tf.logical_not(p)) for x, p in zip(ious, positive_ious)]
        labels = [tf.where(pos_iou, label, float(efficient_det.IGNORE_LABEL)) for pos_iou, label in zip(positive_ious, labels)]
        labels = [tf.where(neg_iou, float(efficient_det.NO_CLASS_LABEL), label) for neg_iou, label in zip(negative_ious, labels)]
        offset = tuple([tf.concat([label[..., None], offset], axis=-1) for label, offset in zip(labels, offsets)])
        return offset

    def _forced_matches(self, n_boxes, matched_boxes, ious):
        # get boxes without a match
        boxes_with_match, _ = tf.unique(tf.concat([box[iou > self.pos_iou_thresh] for iou, box in zip(ious, matched_boxes)], axis=0))
        all_boxes = tf.range(n_boxes, dtype=tf.int64)
        m = tf.logical_not(tf.reduce_any(all_boxes[:, None] == boxes_with_match[None], axis=-1))
        boxes_without_match = all_boxes[m]

        all_matched = tf.shape(boxes_without_match)[-1] == 0
        extra_match = tf.cond(
            all_matched,
            lambda: Dataset._empty_forced_offset(ious),
            lambda: self._non_empty_forced_offset(n_boxes, boxes_without_match, matched_boxes, ious))
        return extra_match

    def _empty_offset(self, height, width):
        offset = []
        for i in range(NUM_LEVELS):
            stride = level_to_stride(i)
            empty_level = tf.ones([height // stride, width // stride, len(self.anchors.aspects), 5]) * efficient_det.IGNORE_LABEL
            offset.append(empty_level)
        return tuple(offset)

    @staticmethod
    def _empty_forced_offset(ious):
        return [tf.zeros_like(iou, dtype=tf.bool) for iou in ious]

    def _non_empty_forced_offset(self, n_boxes, boxes_without_match, matched_boxes, ious):
        # for each anchor scale, find the maximum overlapped box iou
        max_vals = []
        iou_per_box_at_each_level = []
        for box, iou in zip(matched_boxes, ious):
            mask = box[..., None] == boxes_without_match[None, None, None]
            box_ind = tf.argmax(mask, axis=-1)[..., None]
            iou = tf.where(tf.reduce_any(mask, axis=-1), iou, -100)

            shape = tf.shape(box)
            height = shape[0]
            width = shape[1]
            n_anchors = shape[2]
            xs = tf.range(width, dtype=tf.int64)
            ys = tf.range(height, dtype=tf.int64)
            anchors = tf.range(n_anchors, dtype=tf.int64)
            # todo check is right
            grid_loc = tf.stack(tf.meshgrid(ys, xs, anchors, indexing='ij'), axis=-1)

            indices = tf.concat([grid_loc, box_ind], axis=-1)
            iou_per_box = tf.scatter_nd(indices, iou, (height, width, n_anchors, n_boxes))
            iou_per_box_at_each_level.append(iou_per_box)
            max_val = tf.reduce_max(iou_per_box, [0, 1, 2])
            max_vals.append(max_val)

        max_vals = tf.stack(max_vals)
        max_vals = tf.reduce_max(max_vals, axis=0)
        max_vals = tf.where(max_vals <= 0, 100., max_vals)
        extra_matches = []
        for iou_per_box in iou_per_box_at_each_level:
            is_over = iou_per_box >= max_vals[None, None, None]
            extra_matches.append(tf.reduce_any(is_over, axis=-1))
        
        return extra_matches

    def _raw_validation_set(self,):
        raise NotImplemented

    def _raw_training_set(self,):
        raise NotImplemented

