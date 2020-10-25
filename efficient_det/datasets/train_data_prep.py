import tensorflow as tf

import efficient_det.geometry.box
import efficient_det.model


def unnormalise(f):
    def deco(self, image, bbox, labels):
        bbox = efficient_det.geometry.box.Boxes.from_image_and_boxes(image, bbox)
        bbox._unnormalise()

        image, bbox_unnormalised, labels = f(self, image, bbox.box_tensor, labels)

        return image, bbox_unnormalised, labels
    return deco


class ImageBasicPreparation:

    def __init__(self, min_scale, max_scale, target_shape):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_shape = target_shape

    @unnormalise
    def scale_and_random_crop_normalised(self, image, bbox, labels):
        return self.scale_and_random_crop_unnormalised(image, bbox, labels)

    @tf.function
    def scale_and_random_crop_unnormalised(self, image, bbox, labels):
        """bboxes must be unnormalised!"""
        image, bbox = self._random_scale_image(image, bbox)
        image, bbox = self._pad_to_target_if_needed(image, bbox)
        image, bbox, labels = self._random_crop(image, bbox, labels)
        return image, bbox, labels

    def _random_scale_image(self, image, bbox):
        scale = tf.random.uniform([], self.min_scale, self.max_scale)
        image_hw = ImageBasicPreparation._image_hw(image)
        scaled_shape = tf.cast(scale*tf.cast(image_hw, tf.float32), tf.int32)
        bbox = tf.cast(tf.cast(bbox, tf.float32)*scale, tf.int32)
        return tf.image.resize(image, scaled_shape), bbox

    def _random_crop(self, image, bboxes, labels):
        tlbr = self._crop_tlbr_box(image)
        image = ImageBasicPreparation._crop_image(image, tlbr)
        bboxes, labels = ImageBasicPreparation._crop_bboxes(bboxes, labels, tlbr)
        return image, bboxes, labels

    @staticmethod
    def _crop_image(image, tlbr):
        return image[tlbr[0]:tlbr[2], tlbr[1]:tlbr[3]]

    @staticmethod
    def _crop_bboxes(bboxes, labels, tlbr):
        reduced_boxes = efficient_det.geometry.box.Boxes.intersecting_boxes(tlbr[None], bboxes)[0]
        valid_boxes = efficient_det.geometry.box.Boxes.box_area(reduced_boxes) > 0

        offset = tlbr[:2]
        reduced_boxes -= tf.concat([offset, offset], axis=0)[None]

        reduced_boxes = tf.boolean_mask(reduced_boxes, valid_boxes)
        reduced_labels = tf.boolean_mask(labels, valid_boxes)
        return reduced_boxes, reduced_labels

    def _crop_tlbr_box(self, image):
        image_dims = ImageBasicPreparation._image_hw(image)
        can_choose = image_dims - self.target_shape + 1
        top = tf.random.uniform([1], minval=0, maxval=can_choose[0], dtype=tf.int32)
        left = tf.random.uniform([1], minval=0, maxval=can_choose[1], dtype=tf.int32)
        tl = tf.concat([top, left], axis=0)
        return tf.concat([tl, tl + self.target_shape], axis=0)

    def _pad_to_target_if_needed(self, image, bbox):
        to_pad, bbox_offset = self._calc_padding(image)
        image = tf.pad(image, to_pad)
        bbox = bbox + bbox_offset
        return image, bbox

    def _calc_padding(self, image):
        image_dims = ImageBasicPreparation._image_hw(image)
        to_pad = tf.maximum(self.target_shape - image_dims, 0)
        to_pad = tf.concat([to_pad, tf.constant([0])], axis=0)
        to_pad_0 = to_pad // 2
        to_pad_1 = to_pad - to_pad_0

        bbox_offset = tf.concat([to_pad_0[:2], to_pad_0[:2]], axis=0)
        to_pad = tf.stack([to_pad_0, to_pad_1], axis=-1)
        return to_pad, bbox_offset[None]

    @staticmethod
    def _image_hw(image):
        return tf.shape(image)[:2]