import efficient_det
import tensorflow as tf


def level_to_stride(level):
    return 2 ** (efficient_det.STARTING_LEVEL + level)


def pixel_coordinates(height, width):
    xs = tf.range(width, dtype=tf.float32)
    ys = tf.range(height, dtype=tf.float32)
    return tf.stack(tf.meshgrid(xs, ys), axis=-1)


