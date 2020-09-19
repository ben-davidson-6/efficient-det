import efficient_det.model as model
import efficient_det.datasets.coco as coco
import efficient_det.datasets.train_data_prep as train_data_prep
import tensorflow as tf

# todo add some seeding functionality to reproduce
#      add evaluations both summaries and final
#      setup a keras.fit pipeline basically

# anchors
# todo add octaves between levels
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
# todo add downsampling layer
phi = 0
num_classes = 80
network = model.EfficientDetNetwork(phi, num_classes, anchors.num_boxes())

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
# todo add augmentations
#      tune the maps,
#      DEAL WITH NO LABELS!!!

prepper = train_data_prep.ImageBasicPreparation(min_scale=0.1, max_scale=1.5, target_shape=128)
dataset = coco.Coco(
    anchors=anchors,
    augmentations=None,
    basic_training_prep=prepper,
    batch_size=2)


