import tensorflow as tf

from efficient_det.geometry.box import TLBRBoxes


class ImageBasicPreparation:

    def __init__(self, overlap_percentage, min_scale, max_scale, target_shape):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_shape = target_shape
        self.overlap_percantage = overlap_percentage

    def scale_and_random_crop(self, image, bbox, labels):
        """bboxes must be unnormalised!"""
        image = self._random_scale_image(image)
        image, bbox = self._pad_to_target_if_needed(image, bbox)
        image, bbox, labels = self._random_crop(image, bbox, labels)
        image.set_shape([self.target_shape, self.target_shape, 3])
        return image, bbox, labels

    def _random_scale_image(self, image,):
        scale = tf.random.uniform([], self.min_scale, self.max_scale)
        image_hw = ImageBasicPreparation._image_hw(image)
        scaled_shape = tf.cast(scale*tf.cast(image_hw, tf.float32), tf.int32)
        return tf.image.resize(image, scaled_shape)

    def _random_crop(self, image, bboxes, labels):
        image_dims = ImageBasicPreparation._image_hw(image)
        crop_tlbr = self._crop_tlbr_box(image)
        image = image[crop_tlbr[0]:crop_tlbr[2], crop_tlbr[1]:crop_tlbr[3]]
        bboxes, labels = self._crop_bboxes(bboxes, labels, crop_tlbr, image_dims)
        return image, bboxes.get_tensor(), labels

    def _crop_bboxes(self, bboxes, labels, crop_tlbr, image_dim):
        boxes = TLBRBoxes(bboxes)
        boxes.unnormalise(image_dim[0], image_dim[1])
        crop_tlbr = tf.cast(crop_tlbr, tf.float32)
        cropped_image_tlbr_box = TLBRBoxes(crop_tlbr)
        cropped_boxes = boxes.intersecting_boxes(cropped_image_tlbr_box)

        offset = crop_tlbr[:2]
        cropped_boxes.add_offset(-offset)
        valid_boxes = (cropped_boxes.box_area()/boxes.box_area()) > self.overlap_percantage
        cropped_boxes.normalise(self.target_shape, self.target_shape)
        cropped_boxes = cropped_boxes.boolean_mask(valid_boxes)
        cropped_labels = tf.boolean_mask(labels, valid_boxes)
        return cropped_boxes, cropped_labels

    def _crop_tlbr_box(self, image):
        image_dims = ImageBasicPreparation._image_hw(image)
        can_choose = image_dims - self.target_shape + 1
        top = tf.random.uniform([1], minval=0, maxval=can_choose[0], dtype=tf.int32)
        left = tf.random.uniform([1], minval=0, maxval=can_choose[1], dtype=tf.int32)
        tl = tf.concat([top, left], axis=0)
        crop_box = tf.concat([tl, tl + self.target_shape], axis=0)
        return crop_box

    def _pad_to_target_if_needed(self, image, bbox):
        image_dims_original = ImageBasicPreparation._image_hw(image)
        to_pad = tf.maximum(self.target_shape - image_dims_original, 0)
        to_pad = tf.concat([to_pad, tf.constant([0])], axis=0)
        to_pad_0 = to_pad // 2
        to_pad_1 = to_pad - to_pad_0
        to_pad = tf.stack([to_pad_0, to_pad_1], axis=-1)
        image = tf.pad(image, to_pad, mode='CONSTANT', constant_values=tf.reduce_mean(image))

        # need to incorporate the padding for the boxes
        boxes = TLBRBoxes(bbox)
        boxes.unnormalise(image_dims_original[0], image_dims_original[1])
        to_pad_0 = tf.cast(to_pad_0[:2], tf.float32)
        boxes.add_offset(to_pad_0)
        image_dims_new = ImageBasicPreparation._image_hw(image)
        boxes.normalise(image_dims_new[0], image_dims_new[1])
        return image, boxes.tensor

    @staticmethod
    def _image_hw(image):
        return tf.shape(image)[:2]