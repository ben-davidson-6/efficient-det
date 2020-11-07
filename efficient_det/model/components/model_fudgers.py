import tensorflow as tf
import efficient_det


class DownsampleConv(tf.keras.layers.Layer):
    def __init__(self, depth):
        super(DownsampleConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(depth, kernel_size=2, strides=2, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training)
        return x


class Downsampler(tf.keras.layers.Layer):
    def __init__(self, depth, n_extra):
        super(Downsampler, self).__init__()
        self.downsamplers = [DownsampleConv(depth) for _ in range(n_extra)]
        self.downsamplers = tuple(self.downsamplers)

    def call(self, x, training=None):
        y = x[-1]
        for downsample in self.downsamplers:
            y = downsample(y, training)
            x.append(y)
        return x


class ChannelNormConv(tf.keras.layers.Layer):
    def __init__(self, depth):
        super(ChannelNormConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(depth, kernel_size=1, strides=1, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training)
        return x


class ChannelNormaliser(tf.keras.layers.Layer):
    def __init__(self, depth, n_extra):
        super(ChannelNormaliser, self).__init__()
        self.convs = [ChannelNormConv(depth) for _ in range(efficient_det.STARTING_LEVEL + n_extra)]

    def call(self, x, training=None):
        y = []
        for i, c in zip(x, self.convs):
            y.append(c(i, training))
        return y


