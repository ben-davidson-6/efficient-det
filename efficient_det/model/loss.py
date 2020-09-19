import tensorflow as tf

from efficient_det import NO_CLASS_LABEL


class SampleWeightCalculator:
    def __init__(self, proportion_negative, n_classes):
        """
        Parameters
        ----------
        proportion_negative : a value of 2 eg means twice as many negatives as postives
        """
        self.proportion_negative = proportion_negative
        self.n_classes = n_classes

    def mask_and_update_class(self, y_true, y_pred):
        """
        todo this could be made fancier by sampling according to y_pred
          end up doing some kind of hard negative mining, atm we dont
          use y_pred at all, could also normalise the weight of the mask
        Parameters
        ----------
        y_true: [b, h, w, n_anchor]
        y_pred: [b, h, w, n_anchor, n_class]

        Returns
        -------
        """

        positives = y_true != NO_CLASS_LABEL
        negatives = self.get_negatives(positives)
        sample_weight = SampleWeightCalculator.sample_weight(positives, negatives)
        y_true = self.update_classes(negatives, y_true)
        return sample_weight, y_true

    def get_negatives(self, positives):
        ratio = self.ratio_of_all_negatives_to_take(positives)
        uniform = tf.random.uniform(tf.shape(positives), )
        negatives = tf.logical_and(uniform < ratio, tf.logical_not(positives))
        return negatives

    def ratio_of_all_negatives_to_take(self, positives):
        """return a ratio so that if we pick that many we will have the right balance"""
        num_positive = tf.math.count_nonzero(positives, dtype=tf.int32)
        num_neg = tf.size(positives)
        num_desired_neg = self.proportion_negative*num_positive
        ratio = tf.cast(num_desired_neg/num_neg, tf.float32)
        return ratio

    def update_classes(self, negatives, y_true):
        """NO_CLASS_LABEL is -1 and we have the sample weight to ignore all others"""
        tf.where(negatives, self.n_classes + 1, y_true)
        return y_true

    @staticmethod
    def sample_weight(positive, negative):
        return tf.cast(tf.logical_or(positive, negative), tf.float32)


class EfficientDetLoss(tf.keras.losses.Loss):
    def __init__(self, focal_loss, box_loss, weights, n_classes, sample_weight_calculator):
        super(EfficientDetLoss, self).__init__(name='efficient_det_loss')
        self.focal_loss = focal_loss
        self.box_loss = box_loss
        self.weights = weights
        self.n_classes = n_classes
        self.sample_weight_calculator = sample_weight_calculator

    def call(self, y_true, y_pred):
        """
        This works on a single level, as in keras.fit if there are multiple outputs then the loss will
        be applied to each in the list and combined!

        Parameters
        ----------
        y_true : tuple of tensors (class, regression)
            class      : [batch, height_i, width_i, n_anchors, n_classes]
            regression : [batch, height_i, width_i, n_anchors, 4]
            and where class == NO_CLASS_LABEL indicates there was no overlapping box
        y_pred : tupl of tensors (class, regresssion
            class      : [batch, height_i, width_i, n_anchors, n_classes]
            regression : [batch, height_i, width_i, n_anchors, 4]
        Returns
        -------

        """
        y_true_class, y_true_regression = y_true
        y_pred_class, y_pred_regression = y_pred
        y_pred_class = tf.nn.sigmoid(y_pred_class)

        sample_weight, y_true_class = self.sample_weight_calculator.mask_and_update_class(y_true_class, y_pred_class)
        fl = self.focal_loss(y_true_class, y_pred_class, sample_weight) * self.weights[0]
        bl = self.box_loss(y_true_regression, y_pred_regression, sample_weight) * self.weights[1]
        return fl + bl


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma, n_classes, label_smoothing=0.0):
        super(FocalLoss, self).__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.n_classes = n_classes
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
        y_true = tf.one_hot(y_true, depth=self.n_classes)
        focal_coeff = self.calc_focal_coeff(y_true, y_pred)
        ce = self.cross_entropy(y_true, y_pred, sample_weight=sample_weight)
        return focal_coeff * ce


class BoxRegressionLoss(tf.keras.losses.Loss):
    def __init__(self, delta):
        super(BoxRegressionLoss, self).__init__(name='box_regression')
        self.huber_loss = tf.keras.losses.Huber(delta)

    def call(self, y_true, y_pred, sample_weight=None):
        return self.huber_loss(y_true, y_pred, sample_weight)


def loss(weights, alpha, gamma, delta, n_classes, prop_neg, label_smoothing=0.0):
    fl = FocalLoss(alpha, gamma, n_classes, label_smoothing)
    bl = BoxRegressionLoss(delta=delta)
    sample_calc = SampleWeightCalculator(prop_neg, n_classes)
    return EfficientDetLoss(fl, bl, weights, n_classes, sample_calc)