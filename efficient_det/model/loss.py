import tensorflow as tf
import tensorflow_addons as tfa

from efficient_det import NO_CLASS_LABEL, IGNORE_LABEL
from efficient_det.geometry.box import CentroidWidthBoxes


class EfficientDetLoss(tf.keras.losses.Loss):

    def __init__(self, alpha, gamma, delta, weights, n_classes):
        super(EfficientDetLoss, self).__init__(name='efficient_det_loss')
        self.weights = weights
        self.n_classes = n_classes
        self.delta = delta
        
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred, sample_weight=None):
        """
        This works on a single level, as in keras.fit if there are multiple outputs then the loss will
        be applied to each in the list and combined!

        Parameters
        ----------
        y_true : tensor of shape [batch, height_i, width_i, n_anchors, 1 + 4]
            the first element in the final dimension is the class hackily stored as
            a float and the remaining are regression
        y_pred : tensor of shape [batch, height_i, width_i, n_anchors, n_classes + 4]

        Returns
        -------

        """

        y_true_class, y_true_regression = tf.cast(y_true[..., 0], tf.int32), y_true[..., 1:]
        y_pred_class, y_pred_regression = y_pred[..., :self.n_classes], y_pred[..., self.n_classes:]
        positive_mask, training_mask = EfficientDetLoss._calculate_mask_and_normaliser(y_true_class)

        # class loss
        fl = tfa.losses.sigmoid_focal_crossentropy(
            tf.one_hot(y_true_class, depth=self.n_classes),
            y_pred_class,
            self.alpha,
            self.gamma,
            from_logits=True)

        # box loss regression
        bl = self.huber_loss(y_true_regression, y_pred_regression)

        # add them all together sensibly
        fl = fl * self.weights[0] * training_mask
        bl = bl * self.weights[1] * positive_mask
        reduction_axes = [1, 2, 3]        
        return tf.reduce_sum(fl, axis=reduction_axes) + tf.reduce_sum(bl, axis=reduction_axes)

    def huber_loss(self, y_true, y_pred):
        bl = tf.compat.v1.losses.huber_loss(y_true, y_pred, delta=self.delta, reduction=tf.keras.losses.Reduction.NONE)
        return tf.reduce_sum(bl, axis=-1)

    @staticmethod
    def _calculate_mask_and_normaliser(y_true_class):
        positive_mask = y_true_class >= 0
        positive_mask = tf.cast(positive_mask, tf.float32)
        training_mask = tf.not_equal(y_true_class, IGNORE_LABEL)
        training_mask = tf.cast(training_mask, tf.float32)
        return positive_mask, training_mask


if __name__ == '__main__':
    loss = EfficientDetLoss(0.1, 0., 0.1, tf.constant([1., 1.]), 1)
    print(loss.__dict__)
    # y = tf.random.uniform([2, 2, 2, 2, 1])
    # print(loss(y, y).numpy())