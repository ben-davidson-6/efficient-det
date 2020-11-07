import tensorflow as tf

from efficient_det import NO_CLASS_LABEL


class MeanIOU(tf.keras.metrics.Metric):
    def __init__(self, num_classes):
        self.iou = tf.keras.metrics.MeanIoU(num_classes)
        self.num_classes = num_classes
        super(MeanIOU, self).__init__(name='class_accuracy')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true[..., 0], tf.int64)
        y_pred = tf.nn.sigmoid(y_pred[..., :self.num_classes])
        actual_classes = y_true != NO_CLASS_LABEL
        y_true = tf.boolean_mask(y_true, actual_classes)
        y_pred = tf.boolean_mask(y_pred, actual_classes)

        y_pred = tf.argmax(y_pred, axis=-1)
        self.iou.update_state(y_true, y_pred)

    def reset_states(self):
        self.iou.reset_states()

    def result(self):
        return self.iou.result()

    def get_config(self):
        return {'num_classes': self.num_classes}


if __name__ == '__main__':
    iou = tf.keras.metrics.MeanIoU(3)
    k = tf.random.uniform([10], minval=0, maxval=2, dtype=tf.int32)
    x = tf.random.uniform([10], minval=0, maxval=2, dtype=tf.int32)
    iou.update_state(k, x)
    print(k, x)
    print(iou.result())

