import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import efficient_det.model as model
import efficient_det.datasets.coco as coco
import efficient_det.datasets.train_data_prep as train_data_prep
import tensorflow as tf

# todo
#   build callbacks, probably custom
#       saving
#   add training metrics
#   add some seeding functionality to reproduce
#   add octaves between levels
#   add extra downampling layer
#   add augmentations
#   move to conda and setup environment
#   inference
#       non maximal suppresion
#   the matching takes a long time, we could do this once and for all per network


# anchors
anchor_size = 4
anchor_aspects = [
    (1., 1.),
    (.7, 1.4),
    (1.4, 0.7),
]
iou_match_thresh = 0.3
anchors = model.EfficientDetAnchors(
    anchor_size,
    anchor_aspects,
    num_levels=3,
    iou_match_thresh=iou_match_thresh)

# network
phi = 0
num_classes = 80
efficient_det = model.EfficientDetNetwork(phi, num_classes, anchors)

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
prepper = train_data_prep.ImageBasicPreparation(min_scale=0.5, max_scale=1.5, target_shape=512)
dataset = coco.Coco(
    anchors=anchors,
    augmentations=None,
    basic_training_prep=prepper,
    batch_size=4)

# training loop
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
efficient_det.compile(optimizer=adam, loss=loss)
cbs = [model.TensorboardCallback(dataset.training_set(), dataset.validation_set(), 'logs')]
efficient_det.fit(
    dataset.training_set(),
    #validation_data=dataset.validation_set(),
    epochs=10,
    callbacks=cbs,
    steps_per_epoch=500,
    validation_steps=100,
)
