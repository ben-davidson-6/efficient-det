import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import efficient_det.model as model
import efficient_det.datasets.coco as coco
import efficient_det.datasets.train_data_prep as train_data_prep
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()
# todo
#   deal with the image shape requirements better
#       need to have a way of forcing the shapes to be the same for down and up sampling
#   build callbacks, probably custom
#       saving
#   add better training metrics
#   add some seeding functionality to reproduce
#   add augmentations
#   move to conda and setup environment
#   inference
#       non maximal suppresion
#   profile and optimise
#   the number of levels is a bit esoteric for the anchors

# anchors
anchor_size = 4
anchor_aspects = [
    (1., 1.),
    (.75, 1.5),
    (1.5, 0.75),
]
iou_match_thresh = 0.3
anchors = model.EfficientDetAnchors(
    anchor_size,
    anchor_aspects,
    num_levels=6,
    iou_match_thresh=iou_match_thresh)

# network
phi = 0
num_classes = 80
efficient_det = model.EfficientDetNetwork(phi, num_classes, anchors)

# loss
loss_weights = tf.constant([0.9, 0.1])
alpha = 0.25
gamma = 2.0
delta = 0.1
class_loss = model.FocalLoss(alpha, gamma, num_classes)
box_loss = model.BoxRegressionLoss(delta)
loss = model.EfficientDetLoss(class_loss, box_loss, loss_weights, num_classes)

# dataset
prepper = train_data_prep.ImageBasicPreparation(min_scale=0.5, max_scale=1.5, target_shape=256)
dataset = coco.Coco(
    anchors=anchors,
    augmentations=None,
    basic_training_prep=prepper,
    batch_size=8)

# training loop
adam = tf.keras.optimizers.Adam()
efficient_det.compile(optimizer=adam, loss=loss)
cbs = [model.TensorboardCallback(dataset.training_set(), dataset.validation_set(), 'logs')]
efficient_det.fit(
    dataset.training_set(),
    validation_data=dataset.validation_set(),
    epochs=50,
    callbacks=cbs
)
