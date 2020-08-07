import tensorflow as tf

from efficient_det.model.components.detection_head import number_of_classifications


class Loss(tf.keras.losses.Loss):
    def __init__(self, focal_loss, box_loss, weights, num_classes, num_anchors):
        self.focal_loss = focal_loss
        self.box_loss = box_loss
        self.weights = weights
        self.number_of_classifications = number_of_classifications(num_classes, num_anchors)

    def split_prediction(self, y_pred):
        y_pred_class = y_pred[..., :self.number_of_classifications]
        y_pred_regression = y_pred[..., self.number_of_classifications:]
        return y_pred_class, y_pred_regression

    def call(self, y_true, y_pred):
        y_true_class, y_true_regression = y_true
        y_pred_class, y_pred_regression = self.split_prediction(y_pred)
        fl = self.focal_loss(y_true_class, y_pred_class) * self.weights[0]
        bl = self.box_loss(y_true_regression, y_pred_regression) * self.weights[1]
        return fl + bl


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
        return self.huber_loss(y_true, y_pred, weight)


def loss(num_classes, num_anchors, weights, alpha, gamma, label_smoothing=0.0, normalise=True):
    fl = FocalLoss(alpha, gamma, normalise, label_smoothing)
    bl = BoxRegressionLoss(normalise)
    return Loss(fl, bl, weights, num_classes, num_anchors)