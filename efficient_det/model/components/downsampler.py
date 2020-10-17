import tensorflow as tf


class DownsampleConv(tf.keras.layers.Layer):
    def __init__(self, depth):
        super(DownsampleConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(depth, kernel_size=2, strides=2)
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

        self.first_downsample = tf.keras.layers.Conv2D(depth, kernel_size=2, strides=2)
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        y = x[-1]
        for downsample in self.downsamplers:
            y = downsample(y, training)
            x.append(y)
        return x

