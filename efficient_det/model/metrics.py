import tensorflow as tf

from efficient_det.model.model import PostProcessor
from efficient_det import NO_CLASS_LABEL


class ClassAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, anchors):
        self.anchors = anchors
        self.post_processor = PostProcessor(anchors)
        self.acc = tf.keras.metrics.CategoricalAccuracy()
        self.num_classes = num_classes
        super(ClassAccuracy, self).__init__(name='class_accuracy')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true[..., 0], tf.int32)
        y_pred = tf.nn.sigmoid(y_pred[..., :self.num_classes])
        actual_classes = y_true != NO_CLASS_LABEL
        y_true = tf.boolean_mask(y_true, actual_classes)
        y_pred = tf.boolean_mask(y_pred, actual_classes)
        gt = tf.one_hot(y_true, depth=self.num_classes)
        self.acc.update_state(gt, y_pred)

    def reset_states(self):
        self.acc.reset_states()

    def result(self):
        return self.acc.result()

    def get_config(self):
        return {'num_classes': self.num_classes, 'anchors': self.anchors}


class APAt(tf.keras.metrics.Metric):
    def __init__(self, threshold, anchors):
        super(APAt, self).__init__(name=f'AP@{threshold}')
        self.threshold = threshold
        self.post_processor = PostProcessor(anchors)

    def update_state(self, y_true, y_pred, sample_weight=None):
        flat_tlbr, flat_label, flat_score = self.post_processor.model_out_to_flat_tlbr_label_score(y_pred)
        flat_tlbr_gt, flat_label_gt = self.post_processor.ground_truth_to_flat_tlbr_label(y_true)




    def result(self):
        return self.true_positives

