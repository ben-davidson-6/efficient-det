import pytest
import tensorflow as tf

from efficient_det.model.components.backbone import Backbone


@pytest.mark.skip(reason='Takes a long time to run')
def test_can_build_backbones(valid_model_names):
    for valid_model_name in valid_model_names:
        Backbone(valid_model_name)


@pytest.fixture(scope='module')
def efficientnet_b0():
    model = Backbone('b0')
    yield model


def test_output_features(efficientnet_b0):
    random_image_tensor = tf.random.uniform((1, 256, 256, 3), )
    height = random_image_tensor.shape[2]
    features = efficientnet_b0(random_image_tensor)
    assert len(features) == 3
    k = 8
    for f in features:
        assert f.shape[2] == height//k
        k *= 2


def test_inference_same(efficientnet_b0):
    random_image_tensor = tf.random.uniform((1, 256, 256, 3), )
    out_0 = efficientnet_b0(random_image_tensor, training=False)
    out_1 = efficientnet_b0(random_image_tensor, training=False)
    for o0, o1 in zip(out_0, out_1):
        assert tf.reduce_all(o1 == o0)


def test_train_not_same(efficientnet_b0, random_image_tensor):
    random_image_tensor = tf.random.uniform((1, 256, 256, 3), )
    out_0 = efficientnet_b0(random_image_tensor, training=True)[-1]
    out_1 = efficientnet_b0(random_image_tensor, training=True)[-1]
    assert tf.reduce_any(out_0 != out_1).numpy()

