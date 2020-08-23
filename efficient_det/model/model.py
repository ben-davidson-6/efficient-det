import tensorflow as tf
import math

from efficient_det.model.components.backbone import Backbone
from efficient_det.model.components.bifpn import BiFPN
from efficient_det.model.components.detection_head import DetectionHead


class EfficientDetNetwork(tf.keras.Model):
    def __init__(self, phi, num_classes, num_anchors):
        super(EfficientDetNetwork, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.phi = phi
        self.backbone = self.get_backbone()
        self.bifpn = self.get_bifpn()
        self.detection_head = self.get_detection_head()

    def get_backbone(self):
        return Backbone.application_factory(self.phi)

    def get_bifpn(self):
        depth = int(64*(1.35**self.phi))
        repeats = 3 + self.phi
        return BiFPN(depth, repeats)

    def get_detection_head(self):
        repeats = 3 + int(math.floor(self.phi))
        return DetectionHead(self.num_classes, self.num_anchors, repeats)

    def call(self, x, training=None, mask=None):
        x = self.backbone(x, training)
        x = self.bifpn(x, training)
        x = self.detection_head(x, training)
        return x







