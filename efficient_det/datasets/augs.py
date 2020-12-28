import tensorflow as tf

from tf_image.core.bboxes.flip import flip_left_right, flip_up_down
from tf_image.core.colors import rgb_shift


class Augmenter:

    def __call__(self, image, bboxes, labels):
        flip_lr = tf.random.uniform([]) > 0.5
        image, bboxes = tf.cond(flip_lr, lambda: flip_left_right(
            image, bboxes), lambda: (image, bboxes))
        bboxes = random_jitter_boxes(bboxes)
        image = rgb_shift(image, r_shift=0.1, g_shift=0.1, b_shift=0.1)
        return image, bboxes, labels


def random_jitter_boxes(boxes, ratio=0.05, seed=None):
    def random_jitter_box(box, ratio, seed):
        rand_numbers = tf.random.uniform(
            [1, 1, 4], minval=-ratio, maxval=ratio, dtype=tf.float32, seed=seed)
        box_width = tf.subtract(box[0, 0, 3], box[0, 0, 1])
        box_height = tf.subtract(box[0, 0, 2], box[0, 0, 0])
        hw_coefs = tf.stack([box_height, box_width, box_height, box_width])
        hw_rand_coefs = tf.multiply(hw_coefs, rand_numbers)
        jittered_box = tf.add(box, hw_rand_coefs)
        jittered_box = tf.clip_by_value(jittered_box, 0.0, 1.0)
        return jittered_box

    # boxes are [N, 4]. Lets first make them [N, 1, 1, 4]
    boxes_shape = tf.shape(boxes)
    boxes = tf.expand_dims(boxes, 1)
    boxes = tf.expand_dims(boxes, 2)
    distorted_boxes = tf.map_fn(lambda x: random_jitter_box(x, ratio, seed), boxes)
    distorted_boxes = tf.reshape(distorted_boxes, boxes_shape)
    return distorted_boxes
