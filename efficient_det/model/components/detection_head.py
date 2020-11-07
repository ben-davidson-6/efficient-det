import tensorflow as tf
import math
import random


class DetectionHead(tf.keras.layers.Layer):
    def __init__(self, num_class, num_anchors, repeats, depth=32):
        super(DetectionHead, self).__init__()
        self.num_classifications = num_class * num_anchors
        self.num_anchors = num_anchors
        self.num_anchor_regressions = num_anchors * 4
        self.fully_connected_head = FullyConnectedHead(
            num_class,
            num_anchors,
            repeats,
            depth)

    def call(self, inputs, training=None):
        """Outputs [b, h, w, n_anchor, 4 + num_class]"""
        outputs = []
        mixed_outputs = self.fully_connected_head(inputs, training)
        for output in mixed_outputs:
            outputs.append(self._reshape_predictions_to_each_anchor(output))
        return outputs

    def _reshape_predictions_to_each_anchor(self, tensor):
        """takes tensor with final dimension self.num_anchors*x -> self.num_anchors, x"""
        return tf.stack(tf.split(tensor, num_or_size_splits=self.num_anchors, axis=-1), axis=-2)


class FullyConnectedHead(tf.keras.layers.Layer):

    def __init__(self, num_class, num_anchors, repeats, depth):
        super(FullyConnectedHead, self).__init__()
        self.repeats = repeats
        self.depth = depth
        self.convs = [self.seperable_conv_layer() for _ in range(self.repeats)]
        self.bns = [tf.keras.layers.BatchNormalization() for _ in range(self.repeats)]
        number_out = num_anchors*(num_class + 4)
        self.classification_layer = tf.keras.layers.Conv2D(
            filters=number_out,
            kernel_size=1,
            use_bias=True,
            bias_initializer=DetectionBiasInitialiser(num_class, num_anchors),
            dtype=tf.float32)

    def call(self, inputs, training=None):
        outputs = []
        for level in inputs:
            outputs.append(self.put_level_through_head(level, training))
        return outputs

    def put_level_through_head(self, x, training):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x, training)
        x = self.classification_layer(x)
        return x

    def seperable_conv_layer(self):
        return tf.keras.layers.SeparableConv2D(
            self.depth,
            kernel_size=3,
            use_bias=False,
            padding='SAME',
            activation='swish',
            dtype=tf.float32,
        )


class DetectionBiasInitialiser(tf.initializers.Initializer):
    PRIOR_FOREGROUND = 0.01

    def __init__(self, num_class, num_anchor):
        self.initialisation = []
        pi = DetectionBiasInitialiser.PRIOR_FOREGROUND
        prior = -math.log((1 - pi) / pi)
        for anchor in range(num_anchor):
            for c in range(num_class):
                self.initialisation.append(prior)
            self.initialisation += [0. for _ in range(4)]
        self.initialisation = tf.constant(self.initialisation, dtype=tf.float32)

    def __call__(self, shape, dtype=None):
        return self.initialisation

