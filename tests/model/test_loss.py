import pytest
import tensorflow as tf
import efficient_det.model.loss as loss

from pytest_mock import mocker


@pytest.fixture
def learnable_model_and_data():
    class W(tf.keras.models.Model):
        def __init__(self, h, w):
            super(W, self).__init__()
            self.w = tf.Variable(initial_value=tf.ones((1, h, w, 1, 5)))

        def call(self, x, training=None):
            return self.w

    class_inputs = tf.data.Dataset.from_tensor_slices((tf.range(1), -1*tf.ones([1, 1, 1, 1, 1, 5]))).repeat(500)
    return W(1, 1), class_inputs


def test_outputs_correct_shape():
    num_classes = 2
    num_anchors = 1
    l = loss.EfficientDetLoss(alpha=0.25, gamma=1.25, delta=0.1, weights=[0.5, 0.5], n_classes=num_classes)
    y_true = tf.random.uniform((1, 20, 20, num_anchors, 1 + 4), minval=-1, maxval=2, dtype=tf.int32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.random.uniform((1, 20, 20, num_anchors, num_classes + 4), minval=-1, maxval=2, dtype=tf.float32)
    out = l(y_true, y_pred)
    assert out.ndim == 0


def test_doesnt_mask_everything():
    num_classes = 2
    num_anchors = 1
    # do not flip any negatives prop_neg=0 and make box loss 0 with weight
    l = loss.EfficientDetLoss(weights=[.5, 0.5], alpha=0.25, gamma=1.25, delta=0.1, n_classes=num_classes)
    # all classes are negative
    y_true = tf.random.uniform((1, 20, 20, num_anchors, 1 + 4), minval=-1, maxval=2, dtype=tf.int32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.random.uniform((1, 20, 20, num_anchors, num_classes + 4), minval=-1, maxval=2, dtype=tf.float32)

    out = l(y_true, y_pred)
    assert out > 0


def test_loss_can_learn_classes(learnable_model_and_data):
    model, dataset = learnable_model_and_data

    l = loss.EfficientDetLoss(weights=[1., 0.], alpha=0.25, gamma=1.25, delta=0.1, n_classes=1)
    model.compile(tf.keras.optimizers.Adam(.1), loss=l)
    model.fit(dataset)
    weights = model(tf.constant(1))
    learned_weight = tf.nn.sigmoid(weights[..., 0])
    tf.debugging.assert_greater(learned_weight, 0.)
    tf.debugging.assert_less(learned_weight, 0.1)


def test_can_learn_boxes(learnable_model_and_data, mocker):
    model, dataset = learnable_model_and_data

    mocker.patch.object(loss.EfficientDetLoss, attribute='_calculate_mask_and_normaliser', return_value=(1., tf.constant(1.)))
    l = loss.EfficientDetLoss(weights=[0., 1.], alpha=0.25, gamma=1.25, delta=10., n_classes=1)
    model.compile(tf.keras.optimizers.Adam(.1), loss=l)
    model.fit(dataset)
    weights = model(tf.constant(1))
    learned_weight = weights[..., 1:]
    tf.debugging.assert_less(learned_weight, -0.5*tf.ones([4]))
    tf.debugging.assert_greater(learned_weight, -1.2*tf.ones([4]))


def test_can_learn_both(learnable_model_and_data, mocker):
    model, dataset = learnable_model_and_data

    mocker.patch.object(loss.EfficientDetLoss, attribute='_calculate_mask_and_normaliser', return_value=(1., tf.constant(1.)))
    l = loss.EfficientDetLoss(weights=[1., 1.], alpha=0.25, gamma=1.25, delta=10., n_classes=1)
    model.compile(tf.keras.optimizers.Adam(.2), loss=l)
    model.fit(dataset)
    weights = model(tf.constant(1))

    learned_class = tf.nn.sigmoid(weights[..., 0])
    tf.debugging.assert_greater(learned_class, 0.)
    tf.debugging.assert_less(learned_class, 0.1)

    learned_box = weights[..., 1:]
    tf.debugging.assert_less(learned_box, -0.5 * tf.ones([4]))
    tf.debugging.assert_greater(learned_box, -1.2 * tf.ones([4]))




