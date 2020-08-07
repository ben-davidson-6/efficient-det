import tensorflow as tf


def number_of_classifications(num_class, num_anchors):
    """function to make it clear what is goin on in other classes/modules"""
    return num_class*num_anchors


class DetectionHead(tf.keras.layers.Layer):
    def __init__(self, num_class, num_anchors, repeats, dropout_rate=0.2, depth=32):
        super(DetectionHead, self).__init__()
        self.num_classifications = number_of_classifications(num_class, num_anchors)
        self.num_anchors = num_anchors*4
        num_predictions = self.num_classifications + self.num_anchors
        self.fully_connected_head = FullyConnectedHead(num_predictions, repeats, dropout_rate, depth)

    def call(self, inputs, training=None):
        seperated = []
        mixed_outputs = self.fully_connected_head(inputs, training)
        for output in mixed_outputs:
            classifications = output[..., :self.num_classifications]
            regressions = output[..., self.num_classifications:]
            seperated.append((classifications, regressions))
        return seperated


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




