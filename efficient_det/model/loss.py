import tensorflow as tf


class NormalisationMixin:
    def normalization(self, y_true):
        if self.normalize:
            weight = 1./tf.reduce_mean(y_true)
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

    def call(self, y_true, y_pred):
        pred_prob = tf.nn.sigmoid(y_pred)
        focal_coeff = self.calc_focal_coeff(y_true, pred_prob)
        sample_weight = self.normalization(y_true)
        ce = self.cross_entropy(y_true, pred_prob, sample_weight=sample_weight)
        return focal_coeff * ce


class BoxRegressionLoss(NormalisationMixin, tf.keras.losses.Loss):
    def __init__(self, normalize=True):
        super(BoxRegressionLoss, self).__init__(name='box_regression')
        self.huber_loss = tf.keras.losses.Huber(delta=0.1)
        self.normalize = normalize

    @staticmethod
    def pull_out_actual_boxes(self, y_true):
        return tf.not_equal(y_true, 0.0)

    def call(self, y_true, y_pred):
        weight = self.normalization(y_true)
        target_mask = self.actual_target_mask(y_true)
        self.huber_loss(y_true, y_pred, weight)


class BoxIOULoss(tf.keras.losses.Loss):
    pass