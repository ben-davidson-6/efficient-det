import tensorflow as tf


class DetectionHead(tf.keras.layers.Layer):
    def __init__(self, num_class, num_anchors_per_pixel, repeats, dropout_rate=0.2, depth=32):
        super(DetectionHead, self).__init__()
        self.num_classifications = num_class * num_anchors_per_pixel
        self.num_anchors = num_anchors_per_pixel
        self.num_anchor_regressions = num_anchors_per_pixel * 4
        self.fully_connected_head = FullyConnectedHead(
            self.num_classifications,
            self.num_anchor_regressions,
            repeats,
            dropout_rate,
            depth)

    def call(self, inputs, training=None):
        """Outputs [b, h, w, n_anchor, 4 + num_class]"""
        outputs = []
        mixed_outputs = self.fully_connected_head(inputs, training)
        for output in mixed_outputs:
            outputs.append(self._reshape_predictions_to_each_anchor(output))
        return outputs

    def _reshape_output(self, output):
        classifications = output[..., :self.num_classifications]
        regressions = output[..., self.num_classifications:]
        regressions = self._reshape_predictions_to_each_anchor(regressions)
        classifications = self._reshape_predictions_to_each_anchor(classifications)
        return classifications, regressions

    def _reshape_predictions_to_each_anchor(self, tensor):
        """takes tensor with final dimension self.num_anchors*x -> self.num_anchors, x"""
        return tf.stack(tf.split(tensor, num_or_size_splits=self.num_anchors, axis=-1), axis=-2)


class FullyConnectedHead(tf.keras.layers.Layer):

    def __init__(self, num_classifications, num_regressions, repeats, dropout_rate, depth):
        super(FullyConnectedHead, self).__init__()
        self.repeats = repeats
        self.depth = depth
        self.convs = [self.seperable_conv_layer() for _ in range(self.repeats)]
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.classification_layer = tf.keras.layers.Conv2D(
            filters=num_classifications + num_regressions,
            kernel_size=1,
            use_bias=True,
            bias_initializer=DetectionBiasInitialiser(num_classifications, num_regressions),
            dtype=tf.float32)

    def call(self, inputs, training=None):
        outputs = []
        for level in inputs:
            outputs.append(self.put_level_through_head(level))
        return outputs

    def put_level_through_head(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.classification_layer(x)
        return x

    def seperable_conv_layer(self):
        return tf.keras.layers.SeparableConv2D(
            self.depth,
            kernel_size=3,
            padding='SAME',
            activation='relu',
            pointwise_initializer=tf.initializers.VarianceScaling(),
            depthwise_initializer=tf.initializers.VarianceScaling(),
            dtype=tf.float32
        )


class DetectionBiasInitialiser(tf.initializers.Initializer):
    PRIOR_FOREGROUND = 0.01

    def __init__(self, num_classifications, num_regressions):
        self.num_classifications = num_classifications
        self.num_regressions = num_regressions

    def __call__(self, shape, dtype=None):
        class_bias = tf.ones([self.num_classifications])
        pi = DetectionBiasInitialiser.PRIOR_FOREGROUND
        class_bias *= -tf.math.log((1 - pi) / pi)
        bias_initialiser = tf.concat([class_bias, tf.zeros([self.num_regressions])], axis=0)
        return bias_initialiser

