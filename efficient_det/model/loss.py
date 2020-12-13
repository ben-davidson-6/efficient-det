import tensorflow as tf

from efficient_det import NO_CLASS_LABEL


class EfficientDetLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma, delta, weights, n_classes):
        super(EfficientDetLoss, self).__init__(name='efficient_det_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.n_classes = n_classes
        self.delta = delta

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
        non_background, num_positive = EfficientDetLoss._calculate_mask_and_normaliser(y_true_class)
        fl = self.focal_loss(y_true_class, y_pred_class) * self.weights[0] / num_positive[..., None]
        bl = self.huber_loss(y_true_regression, y_pred_regression) * self.weights[1] * non_background / num_positive
        return fl + bl

    def huber_loss(self, y_true, y_pred):
        print(y_true, y_pred)
        return tf.compat.v1.losses.huber_loss(y_true, y_pred, delta=self.delta)

    @staticmethod
    def _calculate_mask_and_normaliser(y_true_class):
        non_background = y_true_class != NO_CLASS_LABEL
        non_background = tf.cast(non_background, tf.float32)
        num_positive_per_image = tf.maximum(tf.reduce_sum(non_background, [1, 2, 3], keepdims=True), 1.)
        return non_background, num_positive_per_image

    def _prep_inputs_focal_loss(self, y_true, y_pred):
        y_true = tf.one_hot(y_true, depth=self.n_classes)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        return y_true, y_pred

    def focal_loss(self, y_true, y_pred):
        y_pred_probs = tf.nn.sigmoid(y_pred)
        y_true, y_pred_probs = self._prep_inputs_focal_loss(y_true, y_pred_probs)
        p_t = y_true*y_pred_probs + (1. - y_true)*(1. - y_pred_probs)
        alpha_factor = tf.ones_like(y_true) * self.alpha
        alpha_t = y_true*alpha_factor + (1. - y_true)*(1. - alpha_factor)
        weight = alpha_t * (1. - p_t)**self.gamma
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return ce*weight
