import tensorflow as tf
import efficient_det


class ClassPrecision(tf.keras.metrics.Metric):
    def __init__(self, num_classes):
        self.precision = tf.keras.metrics.Precision()
        self.num_classes = num_classes
        super(ClassPrecision, self).__init__(name='precision')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true[..., 0], tf.int64)
        mask = efficient_det.IGNORE_LABEL != y_true
        y_true = tf.one_hot(y_true, depth=self.num_classes)
        y_pred = tf.nn.sigmoid(y_pred[..., :self.num_classes])
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        self.precision.update_state(y_true, y_pred)

    def reset_states(self):
        self.precision.reset_states()

    def result(self):
        return self.precision.result()

    def get_config(self):
        return {'num_classes': self.num_classes}


class ClassRecall(tf.keras.metrics.Metric):
    def __init__(self, num_classes):
        self.recall = tf.keras.metrics.Recall()
        self.num_classes = num_classes
        super(ClassRecall, self).__init__(name='recall')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true[..., 0], tf.int64)
        mask = efficient_det.IGNORE_LABEL != y_true
        y_true = tf.one_hot(y_true, depth=self.num_classes)
        y_pred = tf.nn.sigmoid(y_pred[..., :self.num_classes])
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        self.recall.update_state(y_true, y_pred)

    def reset_states(self):
        self.recall.reset_states()

    def result(self):
        return self.recall.result()

    def get_config(self):
        return {'num_classes': self.num_classes}
    
    


class AverageOffsetDiff(tf.keras.metrics.Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.avg_off = tf.keras.metrics.Mean()
        super(AverageOffsetDiff, self).__init__(name='offset_err')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(y_true[..., 0], tf.int64) >= 0
        offset_gt = tf.boolean_mask(y_true[..., 1:3], mask)
        offset_pred = tf.boolean_mask(y_pred[..., self.num_classes:self.num_classes + 2], mask)
        offset_err = tf.reduce_mean(tf.losses.mean_absolute_error(offset_gt, offset_pred, ))
        offset_err = tf.cond(tf.math.is_nan(offset_err), lambda: self.avg_off.result(), lambda: offset_err)
        self.avg_off.update_state(offset_err)

    def reset_states(self):
        self.avg_off.reset_states()

    def result(self):
        return self.avg_off.result()

    def get_config(self):
        return {'num_classes': self.num_classes}


class AverageScaleDiff(tf.keras.metrics.Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.avg_off = tf.keras.metrics.Mean()
        super(AverageScaleDiff, self).__init__(name='scale_err')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(y_true[..., 0], tf.int64) >= 0
        scale_gt = tf.exp(tf.boolean_mask(y_true[..., 3:], mask))
        scale_pred = tf.exp(tf.boolean_mask(y_pred[..., self.num_classes + 2:], mask))
        offset_err = tf.reduce_mean(tf.losses.mean_absolute_error(scale_gt, scale_pred, ))
        offset_err = tf.cond(tf.math.is_nan(offset_err), lambda: self.avg_off.result(), lambda: offset_err)
        self.avg_off.update_state(offset_err)

    def reset_states(self):
        self.avg_off.reset_states()

    def result(self):
        return self.avg_off.result()

    def get_config(self):
        return {'num_classes': self.num_classes}



if __name__ == '__main__':
    iou = tf.keras.metrics.MeanIoU(3)
    k = tf.random.uniform([10], minval=0, maxval=2, dtype=tf.int32)
    x = tf.random.uniform([10], minval=0, maxval=2, dtype=tf.int32)
    iou.update_state(k, x)
    print(k, x)
    print(iou.result())

