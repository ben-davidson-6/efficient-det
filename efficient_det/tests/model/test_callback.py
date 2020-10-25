import pytest
import tensorflow as tf

import efficient_det.model.callbacks as callbacks
import efficient_det.datasets.coco as coco
import efficient_det.model.anchor as anchors
import efficient_det.datasets.train_data_prep as train_data_prep
import efficient_det.model as model


@pytest.fixture
def dataset_and_compiled_model():
    a = anchors.EfficientDetAnchors(4, [(1., 1.)], num_levels=3, iou_match_thresh=0.1)
    prepper = train_data_prep.ImageBasicPreparation(min_scale=0.1, max_scale=1.5, target_shape=256)
    dataset = coco.Coco(a, None, prepper, batch_size=4)

    # network
    phi = 0
    num_classes = 80
    efficient_det = model.EfficientDetNetwork(phi, num_classes, a)

    #loss
    loss_weights = tf.constant([0.5, 0.5])
    alpha = 0.25
    gamma = 1.5
    delta = 0.1
    class_loss = model.FocalLoss(alpha, gamma, num_classes)
    box_loss = model.BoxRegressionLoss(delta)
    loss = model.EfficientDetLoss(class_loss, box_loss, loss_weights, num_classes)
    efficient_det.compile(optimizer='adam', loss=loss)
    return dataset, efficient_det


def test_draw(dataset_and_compiled_model, plt):
    dataset, model = dataset_and_compiled_model
    tensorboard_callback = callbacks.TensorboardCallback(
        dataset.training_set(),
        dataset.validation_set(),
        'logs'
    )
    tensorboard_callback.model = model
    validation, training = tensorboard_callback._build_summary_image()
    plt.subplot(2, 1, 1)
    plt.imshow(training[0])
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(validation[0])
    plt.axis('off')
    plt.saveas = f"{plt.saveas[:-4]}.png"


@pytest.mark.skip('bklah')
def test_learning(dataset_and_compiled_model):

    dataset, model = dataset_and_compiled_model
    tensorboard_callback = callbacks.TensorboardCallback(
        dataset.training_set(),
        dataset.validation_set(),
        'logs'
    )
    model.fit(
        dataset.training_set().take(1),
        validation_data=dataset.validation_set().take(1),
        epochs=1,
        callbacks=[tensorboard_callback]
    )
