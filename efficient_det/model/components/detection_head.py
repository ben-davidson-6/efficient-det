import tensorflow as tf


class DetectionHead(tf.keras.layers.Layer):
    def __init__(self, num_class, num_anchors_per_pixel, repeats, dropout_rate=0.2, depth=32):
        super(DetectionHead, self).__init__()
        self.num_classifications = num_class * num_anchors_per_pixel
        self.num_anchors = num_anchors_per_pixel
        self.num_anchor_regressions = num_anchors_per_pixel * 4
        num_predictions = self.num_classifications + self.num_anchor_regressions
        self.fully_connected_head = FullyConnectedHead(num_predictions, repeats, dropout_rate, depth)

    def call(self, inputs, training=None):
        seperated = []
        mixed_outputs = self.fully_connected_head(inputs, training)
        for output in mixed_outputs:
            classifications, regressions = self._reshape_output(output)
            seperated.append((classifications, regressions))
        return seperated

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
    def __init__(self, number_predictions, repeats, dropout_rate, depth):
        super(FullyConnectedHead, self).__init__()
        self.repeats = repeats
        self.depth = depth
        self.convs = [self.seperable_conv_layer() for _ in range(self.repeats)]
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.classification_layer = tf.keras.layers.Conv2D(
            filters=number_predictions,
            kernel_size=1,
            use_bias=False)

    def call(self, inputs, training=None):
        outputs = []
        for level in inputs:
            outputs.append(self.put_level_through_head(level, training))
        return outputs

    def put_level_through_head(self, x, training):
        for conv in self.convs:
            x = conv(x)
        x = self.dropout(x, training)
        x = self.classification_layer(x)
        return x

    def seperable_conv_layer(self):
        return tf.keras.layers.SeparableConv2D(
            self.depth,
            kernel_size=3,
            padding='SAME',
            pointwise_initializer=tf.initializers.VarianceScaling(),
            depthwise_initializer=tf.initializers.VarianceScaling(),
        )




