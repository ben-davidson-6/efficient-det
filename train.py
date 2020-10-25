import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import efficient_det.model as model
import efficient_det.datasets.coco as coco
import efficient_det.datasets.train_data_prep as train_data_prep
import tensorflow as tf
import datetime

# todo
#   deal with the image shape requirements better
#       need to have a way of forcing the shapes to be the same for down and up sampling
#   add better training metrics
#       iou
#   add some seeding functionality to reproduce
#   add augmentations
#   move to conda and setup environment
#   inference
#       non maximal suppresion
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
loss_weights = tf.constant([1., 99.])
alpha = 0.9
gamma = 2.0
delta = 0.1
class_loss = model.FocalLoss(alpha, gamma, num_classes)
box_loss = model.BoxRegressionLoss(delta)
loss = model.EfficientDetLoss(class_loss, box_loss, loss_weights, num_classes)

# dataset
prepper = train_data_prep.ImageBasicPreparation(min_scale=0.8, max_scale=1.2, target_shape=512)
dataset = coco.Coco(
    anchors=anchors,
    augmentations=None,
    basic_training_prep=prepper,
    batch_size=4)

# training loop
time = datetime.datetime.utcnow().strftime('%h_%d_%H%M%S')
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
efficient_det.compile(optimizer=adam, loss=loss)
save_model = tf.keras.callbacks.ModelCheckpoint(
    f'./artifacts/models/{time}/model',
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch')
tensorboard_vis = model.TensorboardCallback(dataset.training_set(), dataset.validation_set(), f'./artifacts/logs/{time}')
cbs = [save_model, tensorboard_vis]

efficient_det.fit(
    dataset.training_set().repeat(),
    validation_data=dataset.validation_set().repeat(),
    steps_per_epoch=2000,
    validation_steps=500,
    epochs=2000,
    callbacks=cbs
)
