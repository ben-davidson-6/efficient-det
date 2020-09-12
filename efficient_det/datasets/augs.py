import tensorflow as tf
import efficient_det.model.anchor


class ImageBasicPreparation:
    def __int__(self, min_scale, max_scale, target_shape):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_shape = target_shape

    def scale_and_random_crop(self, image, bbox, labels):
        image = self._random_scale_image(image)
        image, bbox = self._pad_to_target_if_needed(image, bbox)
        image, bbox, labels = self._random_crop(image, bbox, labels)
        return image, bbox, labels

    def _random_scale_image(self, image):
        scale = tf.random.uniform(self.min_scale, self.max_scale)
        scaled_shape = tf.cast(scale*ImageBasicPreparation._image_hw(image), tf.int32)
        scaled_image_dims = tf.concat([scaled_shape, tf.shape(image)[-1:]])
        return tf.image.resize(image, scaled_image_dims)

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
        reduced_boxes = ImageBasicPreparation.intersection(tlbr, bboxes)
        valid_boxes = efficient_det.model.anchor.Boxes.box_area(reduced_boxes) > 0.
        return reduced_boxes[valid_boxes], labels[valid_boxes]

    def _crop_tlbr_box(self, image):
        image_dims = ImageBasicPreparation._image_hw(image)
        can_choose = image_dims - self.target_shape
        tl = tf.random.uniform([2], minval=tf.zeros([2]), maxval=can_choose + 1, dtype=tf.int32)
        return tf.concat([tl, tl + self.target_shape])

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

        bbox_offset = tf.concat([to_pad_0[:2], to_pad_0[:2]])
        to_pad = tf.stack([to_pad_0, to_pad_1], axis=-1)
        return to_pad, bbox_offset[None]

    @staticmethod
    def _image_hw(image):
        return tf.shape(image)[:2]

    @staticmethod
    def intersection(image_box, gt_boxes):
        xmin = tf.maximum(image_box[None], gt_boxes)
        ymin = tf.maximum(image_box[None], gt_boxes)
        xmax = tf.minimum(image_box[None], gt_boxes)
        ymax = tf.minimum(image_box[None], gt_boxes)
        intersecting_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)[0]
        return intersecting_boxes


