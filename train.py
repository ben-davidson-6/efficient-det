import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import efficient_det.model as model
import efficient_det.datasets.coco as coco
import efficient_det.datasets.train_data_prep as train_data_prep
import tensorflow as tf
import datetime

# todo
#   deal with the image shape requirements better
#       need to have a way of forcing the shapes to be the same for down and up sampling
#   add better training metrics
#       ap
#       per class accuracy/dice/conf
#   add some seeding functionality to reproduce
#   add augmentations
#   move to conda and setup environment
#   inference
#       non maximal suppresion
#   the number of levels is a bit esoteric for the anchors
#       it has to match up with the downampling of the model

# anchors
anchor_size = 4
base_aspects = [
    (1., 1.),
    (.75, 1.5),
    (1.5, 0.75),
]
aspects = []
for octave in range(3):
    scale = 2**(octave/3)
    for aspect in base_aspects:
        aspects.append((aspect[0]*scale, aspect[1]*scale))

num_levels = 6
anchors = model.build_anchors(anchor_size, num_levels=num_levels, aspects=aspects)

# network
phi = 0
num_classes = 80
efficient_det = model.EfficientDetNetwork(phi, num_classes, anchors)

# loss
loss_weights = tf.constant([1., 1.])
gamma = 2.0
delta = 0.1
alpha = 0.25
class_loss = model.FocalLoss(alpha, gamma, num_classes)
box_loss = model.BoxRegressionLoss(delta)
loss = model.EfficientDetLoss(class_loss, box_loss, loss_weights, num_classes)

# dataset
prepper = train_data_prep.ImageBasicPreparation(min_scale=1.0, max_scale=1, target_shape=256)
iou_match_thresh = 0.5
dataset = coco.Coco(
    anchors=anchors,
    augmentations=None,
    basic_training_prep=prepper,
    iou_thresh=iou_match_thresh,
    batch_size=16)

# training loop
time = datetime.datetime.utcnow().strftime('%h_%d_%H%M%S')
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = [model.ClassAccuracy(num_classes)]
efficient_det.compile(optimizer=adam, loss=loss, metrics=metrics)
save_model = tf.keras.callbacks.ModelCheckpoint(
    f'./artifacts/models/{time}/model',
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch')


tensorboard_vis = model.TensorboardCallback(dataset.training_set(), dataset.validation_set(), f'./artifacts/logs/{time}')
cbs = [tensorboard_vis]
efficient_det.fit(
    dataset.training_set().repeat(),
    # validation_data=dataset.validation_set().repeat(),
    steps_per_epoch=500,
    # validation_steps=1000,
    epochs=999999,
    callbacks=cbs
)
