import tensorflow as tf


class EfficientDetLoss(tf.keras.losses.Loss):
    def __init__(self, focal_loss, box_loss, weights):
        super(EfficientDetLoss, self).__init__(name='efficient_det_loss')
        self.focal_loss = focal_loss
        self.box_loss = box_loss
        self.weights = weights

    def call(self, y_true, y_pred):
        y_true_class, y_true_regression = y_true
        y_pred_class, y_pred_regression = y_pred
        fl = self.focal_loss(y_true_class, y_pred_class) * self.weights[0]
        bl = self.box_loss(y_true_regression, y_pred_regression) * self.weights[1]
        return fl + bl


class NormalisationMixin:
    def normalization(self, y_true):
        if self.normalize:
            weight = tf.math.divide_no_nan(1., tf.reduce_mean(y_true))
        else:
            weight = None
        return weight


class FocalLoss(NormalisationMixin, tf.keras.losses.Loss):
    def __init__(self, alpha, gamma, normalize=True, label_smoothing=0.0):
        super(FocalLoss, self).__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.normalize = normalize

        # this is binary because classes are not mutually exclusive at each
        # point where we predict anchors
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=label_smoothing)
        self.label_smoothin = label_smoothing

    def calc_focal_coeff(self, y_true, pred_prob):
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating = (1.0 - p_t) ** self.gamma
        return alpha * modulating

    def call(self, y_true, y_pred, sample_weight=None):
        pred_prob = tf.nn.sigmoid(y_pred)
        focal_coeff = self.calc_focal_coeff(y_true, pred_prob)
        sample_weight = self.normalization(y_true)
        ce = self.cross_entropy(y_true, pred_prob, sample_weight=sample_weight)
        return focal_coeff * ce


class BoxRegressionLoss(NormalisationMixin, tf.keras.losses.Loss):
    def __init__(self, delta, normalize=True):
        super(BoxRegressionLoss, self).__init__(name='box_regression')
        self.huber_loss = tf.keras.losses.Huber(delta)
        self.normalize = normalize

    def call(self, y_true, y_pred, sample_weight=None):
        weight = self.normalization(y_true)
        return self.huber_loss(y_true, y_pred, weight)


def loss(weights, alpha, gamma, label_smoothing=0.0, normalise=True):
    fl = FocalLoss(alpha, gamma, normalise, label_smoothing)
    bl = BoxRegressionLoss(normalise)
    return EfficientDetLoss(fl, bl, weights)