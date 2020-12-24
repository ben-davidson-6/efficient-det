import tensorflow as tf


class ClassPrecision(tf.keras.metrics.Metric):
    def __init__(self, num_classes):
        # only works for binary
        self.precision = tf.keras.metrics.Precision()
        self.num_classes = num_classes
        super(ClassPrecision, self).__init__(name='precision')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true[..., 0], tf.int64)
        y_true = tf.one_hot(y_true, depth=self.num_classes)
        y_pred = tf.nn.sigmoid(y_pred[..., :self.num_classes])
        self.precision.update_state(y_true, y_pred)

    def reset_states(self):
        self.precision.reset_states()

    def result(self):
        return self.precision.result()

    def get_config(self):
        return {'num_classes': self.num_classes}


if __name__ == '__main__':
    iou = tf.keras.metrics.MeanIoU(3)
    k = tf.random.uniform([10], minval=0, maxval=2, dtype=tf.int32)
    x = tf.random.uniform([10], minval=0, maxval=2, dtype=tf.int32)
    iou.update_state(k, x)
    print(k, x)
    print(iou.result())

