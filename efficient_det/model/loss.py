import tensorflow as tf

from efficient_det import NO_CLASS_LABEL


class EfficientDetLoss(tf.keras.losses.Loss):
    def __init__(self, focal_loss, box_loss, weights, n_classes):
        super(EfficientDetLoss, self).__init__(name='efficient_det_loss')
        self.focal_loss = focal_loss
        self.box_loss = box_loss
        self.weights = weights
        self.n_classes = n_classes

    def call(self, y_true, y_pred):
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
        y_pred_class = tf.nn.sigmoid(y_pred_class)
        fl = self.focal_loss(y_true_class, y_pred_class) * self.weights[0]

        sample_weight_box = tf.cast(y_true_class != NO_CLASS_LABEL, tf.float32)
        bl = self.box_loss(y_true_regression, y_pred_regression, sample_weight_box) * self.weights[1]
        return fl + bl


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma, n_classes):
        super(FocalLoss, self).__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.n_classes = n_classes

    def _prep_inputs(self, y_true, y_pred):
        y_true = tf.one_hot(y_true, depth=self.n_classes)
        eps = tf.keras.backend.epsilon()
        y_pred = y_pred + eps
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        return y_true, y_pred

    def call(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = self._prep_inputs(y_true, y_pred)
        is_on = y_true == 1
        p_t = tf.where(is_on, y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * self.alpha
        alpha_t = tf.where(is_on, alpha_factor, 1 - alpha_factor)
        weight = alpha_t * (1 - p_t) ** self.gamma
        ce = -tf.math.log(p_t)
        return ce*weight


class BoxRegressionLoss(tf.keras.losses.Loss):
    def __init__(self, delta):
        super(BoxRegressionLoss, self).__init__(name='box_regression')
        self.huber_loss = tf.keras.losses.Huber(delta)

    def call(self, y_true, y_pred, sample_weight=None):
        return self.huber_loss(y_true, y_pred, sample_weight)


def loss(weights, alpha, gamma, delta, n_classes):
    fl = FocalLoss(alpha, gamma, n_classes)
    bl = BoxRegressionLoss(delta=delta)
    return EfficientDetLoss(fl, bl, weights, n_classes)