import tensorflow as tf


def draw_model_output(image, boxes, scores, thresh):
    if tf.rank(image) == 3:
        image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.uint8)
    for box, score in zip(boxes, scores):
        to_show = score > thresh
        box = tf.boolean_mask(box, to_show[..., 0])
        image = tf.image.draw_bounding_boxes(tf.image.convert_image_dtype(image, tf.float32), box[None], tf.constant([(0., 0., 1.), (0., 1., 0.), (1., 0., 0.)]))
    return image

