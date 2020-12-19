import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import efficient_det.model as model
import efficient_det.datasets.wider_face as faces
import efficient_det.datasets.train_data_prep as train_data_prep
import tensorflow as tf
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
anchor_size = 2
base_aspects = [
    (1., 1.),
    (.75, 1.5),
    (1.5, 0.75),
]
aspects = []
n_octave = 2
for octave in range(n_octave):
    scale = 2**(octave / n_octave)
    for aspect in base_aspects:
        aspects.append((aspect[0] * scale, aspect[1] * scale))
num_levels = 6
anchors = model.build_anchors(anchor_size, num_levels=num_levels, aspects=aspects)

# network
phi = 0
num_classes = 1
efficient_det = model.EfficientDetNetwork(phi, num_classes, anchors, n_extra_downsamples=3)

# loss
loss_weights = tf.constant([1., 1.])
gamma = 1.5
delta = 0.1
alpha = 0.75
loss = model.EfficientDetLoss(alpha, gamma, delta, loss_weights, num_classes)

# dataset
iou_match_thresh = 0.5
overlap_for_crop = 0.3
prepper = train_data_prep.ImageBasicPreparation(
    min_scale=0.8,
    overlap_percentage=overlap_for_crop,
    max_scale=1.5,
    target_shape=512)

dataset = faces.Faces(
    anchors=anchors,
    augmentations=None,
    basic_training_prep=prepper,
    iou_thresh=iou_match_thresh,
    batch_size=5)

# training loop
time = datetime.datetime.utcnow().strftime('%h_%d_%H%M%S')
adam = tf.keras.optimizers.Adam(learning_rate=0.001,)

metrics = [model.ClassAccuracy(num_classes)]
efficient_det.compile(optimizer=adam, loss=loss, metrics=metrics)
save_best_model = tf.keras.callbacks.ModelCheckpoint(
    f'./artifacts/models/{time}/best_model',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch')
save_most_recent_model = tf.keras.callbacks.ModelCheckpoint(
    f'./artifacts/models/{time}/best_model',
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch')

tensorboard_vis = model.TensorboardCallback(dataset.training_set(), dataset.validation_set(), f'./artifacts/logs/{time}')
cbs = [save_best_model, save_most_recent_model, tensorboard_vis]
efficient_det.fit(
    dataset.training_set(),
    validation_data=dataset.validation_set(),
    epochs=300,
    callbacks=cbs,
    verbose=0
)

