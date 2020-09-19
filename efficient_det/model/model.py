import tensorflow as tf
import math
import pprint
from efficient_det.model.components.backbone import Backbone
from efficient_det.model.components.bifpn import BiFPN
from efficient_det.model.components.detection_head import DetectionHead


class EfficientDetNetwork(tf.keras.Model):
    N_LEVELS = 3

    def __init__(self, phi, num_classes, num_anchors):
        super(EfficientDetNetwork, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.phi = phi
        self.backbone = self.get_backbone()
        self.bifpn = self.get_bifpn()
        self.detection_head = self.get_detection_head()
        self.loss = None
        self.optimizer = None

    def call(self, x, training=None, mask=None):
        x = self.backbone(x, training)
        x = self.bifpn(x, training)
        x = self.detection_head(x, training)
        return x

    def train_step(self, data):
        image, regressions = data
        with tf.GradientTape() as tape:
            pred_regressions = self(image, training=True)
            loss = 0.
            for level in range(EfficientDetNetwork.N_LEVELS):
                y_true = regressions[level]
                y_pred = pred_regressions[level]
                loss += self.loss(
                    y_true,
                    y_pred)
                self.compiled_metrics.update_state(y_true, y_pred)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        image, regressions = data
        # Compute predictions
        pred_regressions = self(image, training=False)
        loss = 0.
        for level in range(EfficientDetNetwork.N_LEVELS):
            y_true = regressions[level]
            y_pred = pred_regressions[level]
            loss += self.loss(
                y_true,
                y_pred)
            # self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer(self, optimiser):
        self.optimizer = optimiser

    def get_backbone(self):
        return Backbone.application_factory(self.phi)

    def get_bifpn(self):
        depth = int(64*(1.35**self.phi))
        repeats = 3 + self.phi
        return BiFPN(depth, repeats)

    def get_detection_head(self):
        repeats = 3 + int(math.floor(self.phi))
        return DetectionHead(self.num_classes, self.num_anchors, repeats)









