import efficient_det.model as model
import tensorflow as tf

# anchors
anchor_size = 4
anchor_aspects = [
    (1., 1.),
    (.7, 1.4),
    (1.4, 0.7),
]
iou_match_thresh = 0.5
anchors = model.EfficientDetAnchors(anchor_size, anchor_aspects, num_levels=3, iou_match_thresh=iou_match_thresh)

# network
phi = 0
num_classes = 80
network = model.EfficientDetNetwork(phi, num_classes, anchors.num_boxes())

# loss
loss_weights = tf.constant([0.5, 0.5])
alpha = 0.25
gamma = 1.5
delta = 0.1
class_loss = model.FocalLoss(alpha, gamma)
box_loss = model.BoxRegressionLoss(delta)
loss = model.EfficientDetLoss(class_loss, box_loss, loss_weights)