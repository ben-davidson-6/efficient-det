import tensorflow as tf
import math

from efficient_det.model.components.backbone import Backbone
from efficient_det.model.components.downsampler import Downsampler
from efficient_det.model.components.bifpn import BiFPN
from efficient_det.model.components.detection_head import DetectionHead


class EfficientDetNetwork(tf.keras.Model):
    N_LEVELS = 3

    def __init__(self, phi, num_classes, anchors, n_extra_downsamples=3):
        super(EfficientDetNetwork, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = anchors.num_boxes()
        self.phi = phi
        self.backbone = self.get_backbone()
        self.downsampler = Downsampler(depth=64, n_extra=n_extra_downsamples)
        self.bifpn = self.get_bifpn()
        self.detection_head = self.get_detection_head()

    def call(self, x, training=None, mask=None):
        # todo add make sure is sufficiently even
        x = self.backbone(x, training)
        x = self.downsampler(x, training)
        x = self.bifpn(x, training)
        x = self.detection_head(x, training)
        [tf.debugging.check_numerics(y, 'Found nans in output of model', name=None) for y in x]
        return x

    def get_backbone(self):
        return Backbone.application_factory(self.phi)

    def get_bifpn(self):
        depth = int(64*(1.35**self.phi))
        repeats = 3 + self.phi
        return BiFPN(depth, repeats)

    def get_detection_head(self):
        repeats = 3 + int(math.floor(self.phi))
        return DetectionHead(self.num_classes, self.num_anchors, repeats)









