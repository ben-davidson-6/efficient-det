import efficient_det.model as model
import efficient_det.datasets.wider_face as faces
import tensorflow as tf
import tensorflow_addons as tfa
import datetime


# anchors
anchor_size = 1.
base_aspects = [
    (1., 1.),
    (0.7, 1.4),
    (1.4, 0.7)
]
aspects = []
n_octave = 3
for octave in range(n_octave):
    scale = 2**(octave / n_octave)
    for aspect in base_aspects:
        aspects.append((aspect[0] * scale, aspect[1] * scale))
anchors = model.build_anchors(
    anchor_size,
    aspects=aspects)

# dataset
pos_iou_match_thresh = 0.5
neg_iou_thresh = 0.4
target_shape = 512 + 128
batch_size = 16

augmenter = model.Augmenter()
dataset = faces.Faces(
    anchors=anchors,
    augmentations=augmenter,
    image_size=target_shape,
    pos_iou_thresh=pos_iou_match_thresh,
    neg_iou_thresh=neg_iou_thresh,
    batch_size=batch_size)

#  background is handled automatically so dont need + 1 here
num_classes = 1

# loss
loss_weights = [1., 1.]
gamma = 2.0
delta = 1.0
alpha = 0.75
loss = model.EfficientDetLoss(alpha, gamma, delta, loss_weights, num_classes)

# training loop
radam = tfa.optimizers.RectifiedAdam(0.001)
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
metrics = [
    model.ClassPrecision(num_classes),
    model.ClassRecall(num_classes),
    model.AverageOffsetDiff(num_classes),
    model.AverageScaleDiff(num_classes)]

# network
phi = 0
efficient_det = model.EfficientDetNetwork(phi, num_classes, anchors)
efficient_det.compile(optimizer=ranger, loss=loss, metrics=metrics)

# callback
time = datetime.datetime.utcnow().strftime('%h_%d_%H%M%S')
tensorboard_vis = model.TensorboardCallback(dataset, f'./artifacts/{time}')
cbs = [tensorboard_vis]

# Train!
efficient_det.fit(
    dataset.training_set(),
    validation_data=dataset.validation_set(),
    epochs=300,
    callbacks=cbs,
    verbose=2)

