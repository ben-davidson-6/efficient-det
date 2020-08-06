import tensorflow as tf
import efficient_det.model.components.bifpn as bifpn


def test_node_output_shape():
    h = w = 128
    depth = 1
    random_image_tensor = tf.random.uniform((1, h, w, 3), )
    random_image_tensor_small = tf.random.uniform((1, h//2, w//2, 3), )

    # test upsamples by 2, with 2 inputs
    upsampling = bifpn.BiFPNNode(num_inputs=2, depth=depth, upsample=True)
    a_t = upsampling([random_image_tensor, random_image_tensor_small], training=True)
    assert a_t.get_shape() == tf.TensorShape((1, h, w, depth))

    # test downsamples by 2, with 2 inputs
    downsampling = bifpn.BiFPNNode(num_inputs=2, depth=depth, upsample=False)
    a_t = downsampling([random_image_tensor_small, random_image_tensor], training=True)
    assert a_t.get_shape() == tf.TensorShape((1, h//2, w//2, depth))

    # test upsamples by 2, with 3 inputs
    upsampling = bifpn.BiFPNNode(num_inputs=3, depth=depth, upsample=True)
    a_t = upsampling([random_image_tensor, random_image_tensor, random_image_tensor_small], training=True)
    assert a_t.get_shape() == tf.TensorShape((1, h, w, depth))

    # test downsamples by 2, with 3 inputs
    downsampling = bifpn.BiFPNNode(num_inputs=3, depth=depth, upsample=False)
    a_t = downsampling([random_image_tensor_small, random_image_tensor_small, random_image_tensor], training=True)
    assert a_t.get_shape() == tf.TensorShape((1, h // 2, w // 2, depth))


def test_node_inference_and_training_different():
    h = w = 128
    depth = 1
    random_image_tensor = tf.random.uniform((1, h, w, 3), )
    random_image_tensor_small = tf.random.uniform((1, h // 2, w // 2, 3), )

    upsampling = bifpn.BiFPNNode(num_inputs=2, depth=depth, upsample=True)
    training0 = upsampling([random_image_tensor, random_image_tensor_small], training=True)
    inference = upsampling([random_image_tensor, random_image_tensor_small], training=False)
    training1 = upsampling([random_image_tensor, random_image_tensor_small], training=True)

    tf.debugging.assert_none_equal(training0, inference)
    tf.debugging.assert_equal(training0, training1)


def test_node_not_nan():
    h = w = 128
    depth = 1
    random_image_tensor = tf.random.uniform((1, h, w, 3), )
    random_image_tensor_small = tf.random.uniform((1, h // 2, w // 2, 3), )

    upsampling = bifpn.BiFPNNode(num_inputs=2, depth=depth, upsample=True)
    training = upsampling([random_image_tensor, random_image_tensor_small], training=True)
    inference = upsampling([random_image_tensor, random_image_tensor_small], training=False)

    tf.debugging.assert_all_finite(training, message='found nan')
    tf.debugging.assert_all_finite(inference, message='found nan')


def test_bifpn_layer():
    h = w = 128
    depth = 1
    tensors = [tf.random.uniform((1, h//2**i, w//2**i, depth), ) for i in range(5)]
    bifpn_layer = bifpn.BiFPNLayer(depth=depth)
    tensors_out = bifpn_layer(tensors, training=False)
    for in_t, out_t in zip(tensors, tensors_out):
        assert in_t.get_shape() == out_t.get_shape(), 'output of layer not shape or order preserving'





