import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import efficient_det.model as model
import efficient_det.datasets.coco as coco
import efficient_det.datasets.train_data_prep as train_data_prep
import tensorflow as tf

# todo
#   add some seeding functionality to reproduce
#   add training metrics
#   add some kind of regression class to simplify passing a tensor around
#   add octaves between levels
#   add extra downampling layer
#   add augmentations
#   move to conda and setup environment
#   build callbacks, probably custom
#       saving
#       tensorboard
#   inference
#       non maximal suppresion


# anchors
anchor_size = 4
anchor_aspects = [
    (1., 1.),
    (.7, 1.4),
    (1.4, 0.7),
]
iou_match_thresh = 0.5
anchors = model.EfficientDetAnchors(
    anchor_size,
    anchor_aspects,
    num_levels=3,
    iou_match_thresh=iou_match_thresh)

# network
phi = 0
num_classes = 80
efficient_det = model.EfficientDetNetwork(phi, num_classes, anchors.num_boxes())

# loss
loss_weights = tf.constant([0.5, 0.5])
alpha = 0.25
gamma = 1.5
delta = 0.1
prop_neg = 2
class_loss = model.FocalLoss(alpha, gamma, num_classes)
box_loss = model.BoxRegressionLoss(delta)
sample_weight_calculator = model.SampleWeightCalculator(prop_neg, num_classes)
loss = model.EfficientDetLoss(class_loss, box_loss, loss_weights, num_classes, sample_weight_calculator)

# dataset
prepper = train_data_prep.ImageBasicPreparation(min_scale=0.1, max_scale=1.5, target_shape=512)
dataset = coco.Coco(
    anchors=anchors,
    augmentations=None,
    basic_training_prep=prepper,
    batch_size=4)

# training loop
adam = tf.keras.optimizers.Adam()
efficient_det.compile(optimizer=adam, loss=loss)
cbs = [tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=False, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)]
efficient_det.fit(
    dataset.training_set(),
    validation_data=dataset.validation_set(),
    epochs=1,
    callbacks=cbs
)
