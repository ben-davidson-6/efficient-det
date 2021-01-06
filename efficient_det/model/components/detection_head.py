import tensorflow as tf
import math
import random


class DetectionHead(tf.keras.layers.Layer):
    def __init__(self, num_class, num_anchors, repeats, n_levels, depth=32):
        super(DetectionHead, self).__init__()
        self.num_classifications = num_class * num_anchors
        self.num_anchors = num_anchors
        self.num_anchor_regressions = num_anchors * 4
        self.box_net = FullyConnectedHead(
            focal_init=False,
            n_out=num_anchors*4,
            repeats=repeats,
            n_levels=n_levels,
            depth=depth,)
        self.class_net = FullyConnectedHead(
            focal_init=True,
            n_out=num_anchors*num_class,
            repeats=repeats,
            n_levels=n_levels,
            depth=depth,)

    def call(self, inputs, training=None):
        """Outputs [b, h, w, n_anchor, 4 + num_class]"""
        outputs = []
        box_outputs = self.box_net(inputs, training)
        class_outputs = self.class_net(inputs, training)
        for box_out, class_out in zip(box_outputs, class_outputs):
            box_out = self._reshape_predictions_to_each_anchor(box_out)
            class_out = self._reshape_predictions_to_each_anchor(class_out)
            output = tf.concat([class_out, box_out], axis=-1)
            outputs.append(output)
        return outputs

    def _reshape_predictions_to_each_anchor(self, tensor):
        """takes tensor with final dimension self.num_anchors*x -> self.num_anchors, x"""
        return tf.stack(tf.split(tensor, num_or_size_splits=self.num_anchors, axis=-1), axis=-2)


class FullyConnectedHead(tf.keras.layers.Layer):

    def __init__(self, focal_init, n_out, repeats, n_levels, depth):
        super(FullyConnectedHead, self).__init__()
        self.repeats = repeats
        self.depth = depth
        self.convs = [self.seperable_conv_layer() for _ in range(self.repeats)]
        initer = DetectionBiasInitialiser(n_out) if focal_init else tf.keras.initializers.Constant()
        self.classification_layer = tf.keras.layers.Conv2D(
            filters=n_out,
            kernel_size=1,
            use_bias=True,
            bias_initializer=initer,
            dtype=tf.float32)
        self.dropout = tf.keras.layers.Dropout(rate=0.05)

    def call(self, inputs, training=None):
        outputs = []
        for level, level_feats in enumerate(inputs):
            outputs.append(self.put_level_through_head(level_feats, level, training))
        return outputs

    def put_level_through_head(self, x, level, training):
        for k, conv in enumerate(self.convs):
            if k != 0:
                x = self.dropout(x)
            x = conv(x)
        x = self.classification_layer(x)
        return x

    def seperable_conv_layer(self):
        return tf.keras.layers.SeparableConv2D(
            self.depth,
            kernel_size=3,
            use_bias=True,
            padding='SAME',
            activation='swish'
        )


class DetectionBiasInitialiser(tf.initializers.Initializer):
    PRIOR_FOREGROUND = 0.05

    def __init__(self, n):
        self.initialisation = []
        pi = DetectionBiasInitialiser.PRIOR_FOREGROUND
        prior = -math.log((1 - pi) / pi)
        self.initialisation = tf.ones([n])*prior + tf.random.uniform([n])*0.01
        
    def __call__(self, shape, dtype=None):
        return self.initialisation

