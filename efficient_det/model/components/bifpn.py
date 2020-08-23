import tensorflow as tf


class BiFPN(tf.keras.layers.Layer):
    def __init__(self, depth, repeats, shared=True):
        super(BiFPN, self).__init__()
        self.depth = depth
        self.repeats = repeats
        self.shared = shared
        self.bifpn_layers = [self.get_bifpn_layer() for _ in range(self.repeats)]

    def call(self, x, training=None):
        for l in self.bifpn_layers:
            x = l(x, training)
        return x

    def get_bifpn_layer(self):
        return BiFPNLayer(self.depth)


class BiFPNLayer(tf.keras.layers.Layer):
    def __init__(self, depth):
        super(BiFPNLayer, self).__init__()
        self.depth = depth
        self.number_intermediate_features = None
        self.descending_nodes = None
        self.ascending_nodes = None

    def call(self, inputs, training=None):
        """inputs are ordered biggest resolution to smallest"""
        descending_outputs = self.descend_path(inputs, training)
        ascending_outputs = self.ascend_path(inputs, descending_outputs, training)
        ascending_outputs.insert(0, descending_outputs[0])
        return ascending_outputs

    def build(self, input_shape):
        self.number_intermediate_features = len(input_shape)
        self.descending_nodes = [self.descending_node() for _ in range(self.number_intermediate_features - 1)]
        self.ascending_nodes = [self.ascending_node() for _ in range(self.number_intermediate_features - 2)]
        self.ascending_nodes.append(self.ascending_node(final=True))

    def descend_path(self, inputs, training):
        descending_outputs = []
        t_0 = inputs[-1]
        for i in range(self.number_intermediate_features - 2, -1, -1):
            t_1 = inputs[i]
            t_0 = self.descending_nodes[i](
                [t_1, t_0],
                training=training)
            descending_outputs.insert(0, t_0)
        return descending_outputs

    def ascend_path(self, inputs, descend_out, training):
        ascending_outputs = []
        input_feature_offset = 1
        t_0 = descend_out[0]
        for i in range(self.number_intermediate_features - 2):
            t_0 = self.ascending_nodes[i](
                [inputs[input_feature_offset + i], descend_out[i + 1], t_0],
                training=training)
            ascending_outputs.append(t_0)
        t_0 = self.ascending_nodes[-1]([inputs[-1], t_0], training=training)
        ascending_outputs.append(t_0)
        return ascending_outputs

    def descending_node(self, ):
        return BiFPNNode(
            num_inputs=2,
            depth=self.depth,
            upsample=True)

    def ascending_node(self, final=False):
        if final:
            node = BiFPNNode(
                num_inputs=2,
                depth=self.depth,
                upsample=False)
        else:
            node = BiFPNNode(
                num_inputs=3,
                depth=self.depth,
                upsample=False)
        return node


class BiFPNNode(tf.keras.layers.Layer):
    eps = 1e-7

    def __init__(self, num_inputs, depth, upsample):
        super(BiFPNNode, self).__init__()
        assert num_inputs >= 2, 'Number of inputs must be >= 2'

        self.upsample = upsample
        self.num_inputs = num_inputs
        self.depth = depth

        self.resampler = self.build_resampler()
        self.fusion_weights = tf.Variable(initial_value=tf.ones([self.num_inputs]))
        self.seperable_conv = tf.keras.layers.SeparableConv2D(depth, kernel_size=1, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(tf.keras.activations.swish)
        self.concater = tf.keras.layers.Concatenate()

    def call(self, inputs, training=None):
        """inputs[-1] should be the node that needs to be increased or decreased"""
        self.assert_recieved_correct_number_inputs(inputs)
        inputs = self.resize_final_input(inputs)
        x = self.weight_and_concat(inputs)
        x = self.concater(x)
        x = self.seperable_conv(x)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x

    def resize_final_input(self, inputs):
        return [i for i in inputs[:-1]] + [self.resampler(inputs[-1])]

    def weight_and_concat(self, inputs):
        normaliser = tf.reduce_sum(self.fusion_weights) + BiFPNNode.eps
        weights = self.fusion_weights / normaliser
        weighted = [weights[i] * x for i, x in enumerate(inputs)]
        return weighted

    def build_resampler(self):
        if self.upsample:
            return tf.keras.layers.UpSampling2D(size=(2, 2))
        else:
            return tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    def assert_recieved_correct_number_inputs(self, inputs):
        msg = f'\n\tYou did not give {self.num_inputs} inputs to layer {self.name}\n\tyou gave {len(inputs)}'
        assert len(inputs) == self.num_inputs, msg






