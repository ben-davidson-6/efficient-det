import os
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64'
import numpy as np
import efficient_det.model as model
import efficient_det.datasets.wider_face as faces
import efficient_det.datasets.train_data_prep as train_data_prep
import tensorflow as tf
import tensorflow_addons as tfa
import datetime

# todo
#   deal with the image shape requirements better
#       need to have a way of forcing the shapes to be the same for down and up sampling
#   add better training metrics
#       ap
#   add some seeding functionality to reproduce
#   add augmentations
#   move to conda and setup environment
#   the number of levels is a bit esoteric for the anchors
#       it has to match up with the downampling of the model

# anchors
anchor_size = 1.
base_aspects = [
    (1., 1.),
    (0.75, 1.5),
    (1.5, 0.75)
]
aspects = []
n_octave = 3
for octave in range(n_octave):
    scale = 2**(octave / n_octave)
    for aspect in base_aspects:
        aspects.append((aspect[0] * scale, aspect[1] * scale))
num_levels = 6
anchors = model.build_anchors(
    anchor_size,
    num_levels=num_levels,
    aspects=aspects)

# network
phi = 0
num_classes = 1
efficient_det = model.EfficientDetNetwork(phi, num_classes, anchors, n_extra_downsamples=3)

# loss
loss_weights = tf.constant([1., 50.])
gamma = 1.5
delta = 0.1
alpha = 0.25
loss = model.EfficientDetLoss(alpha, gamma, delta, loss_weights, num_classes)

# dataset
pos_iou_match_thresh = 0.5
neg_iou_thresh = 0.4
overlap_for_crop = 0.2
prepper = train_data_prep.ImageBasicPreparation(
    min_scale=0.5,
    overlap_percentage=overlap_for_crop,
    max_scale=1.2,
    target_shape=512)
augmenter = model.Augmenter()

dataset = faces.Faces(
    anchors=anchors,
    augmentations=augmenter,
    basic_training_prep=prepper,
    pos_iou_thresh=pos_iou_match_thresh,
    neg_iou_thresh=neg_iou_thresh,
    batch_size=8)

# training loop
time = datetime.datetime.utcnow().strftime('%h_%d_%H%M%S')
radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
metrics = [model.ClassPrecision(num_classes), model.AverageOffsetDiff(num_classes), model.AverageScaleDiff(num_classes)]
efficient_det.compile(optimizer=ranger, loss=loss, metrics=metrics)

tensorboard_vis = model.TensorboardCallback(dataset, f'./artifacts/{time}')
cbs = [tensorboard_vis]
efficient_det.fit(
    dataset.training_set(),
    validation_data=dataset.validation_set(),
    epochs=300,
    callbacks=cbs,
    verbose=0)

