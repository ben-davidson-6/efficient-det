import os
# print('delet me')
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64'
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import math

from efficient_det.model.components.backbone import Backbone
from efficient_det.model.components.model_fudgers import Downsampler, ChannelNormaliser
from efficient_det.model.components.bifpn import BiFPN
from efficient_det.model.components.detection_head import DetectionHead


class EfficientDetNetwork(tf.keras.Model):
    # should be 3 but memory
    base_repeats = 3
    # should be 64 but memory
    base_depth = 64

    def __init__(self, phi, num_classes, anchors, n_extra_downsamples=2):
        # todo could validate num levels here
        super(EfficientDetNetwork, self).__init__()
        self.num_classes = num_classes
        self.n_extra_downsample = n_extra_downsamples
        self.anchors = anchors
        self.num_anchors = len(anchors.aspects)
        self.phi = phi
        self.backbone = self.get_backbone()
        self.downsampler = self.get_downsampler()
        self.channel_normaliser = self.get_channel_normaliser()
        self.bifpn = self.get_bifpn()
        self.detection_head = self.get_detection_head()

    def call(self, x, training=None, mask=None):
        # todo add make sure is sufficiently even

        # training is false to turn off batch norm updates to
        # mean and var always
        x = self.backbone(x, training=False)
        x = self.downsampler(x, training)
        x = self.channel_normaliser(x, training)
        x = self.bifpn(x, training)
        x = self.detection_head(x, training)
        # [tf.debugging.check_numerics(y, 'Found nans in output of model', name=None) for y in x]
        return x

    def get_backbone(self):
        return Backbone.application_factory(self.phi)

    def get_bifpn(self):
        depth = self.get_bifpn_depth()
        repeats = EfficientDetNetwork.base_repeats + self.phi
        return BiFPN(depth, repeats)

    def get_detection_head(self):
        # should be 3
        repeats = EfficientDetNetwork.base_repeats + int(math.floor(self.phi))
        return DetectionHead(self.num_classes, self.num_anchors, repeats)

    def get_downsampler(self):
        return Downsampler(depth=self.get_bifpn_depth(), n_extra=self.n_extra_downsample)

    def get_channel_normaliser(self):
        return ChannelNormaliser(depth=self.get_bifpn_depth(), n_extra=self.n_extra_downsample)

    def get_bifpn_depth(self):
        return int(EfficientDetNetwork.base_depth*(1.35**self.phi))


class PostProcessor:
    def __init__(self, anchors, num_classes):
        self.anchors = anchors
        self.num_classes = num_classes

    def process_output(self, model_out):
        tlbr, probs = self._model_out_to_flat_tlbr_label_score(model_out)
        tlbr, probs, labels, valid_detections = self._apply_nms(tlbr, probs)
        return tlbr, probs, labels, valid_detections

    def _apply_nms(self, tlbr, probs):
        tlbr, probs, labels, valid_detections = tf.image.combined_non_max_suppression(
            tlbr,
            probs,
            max_output_size_per_class=100,
            max_total_size=100,
            score_threshold=0.1)
        labels = tf.cast(labels, tf.int32)
        return tlbr, probs, labels, valid_detections

    def _model_out_to_flat_tlbr_label_score(self, y_pred):
        offset_tensors, class_probabilities = PostProcessor._predicted_unpack(y_pred)
        tlbr_tensor = self.anchors.to_tlbr_tensor(offset_tensors)
        tlbr, probs = self._reshape_for_nms(
            tlbr_tensor,
            class_probabilities)
        return tlbr, probs

    @staticmethod
    def _predicted_unpack(y_pred):
        out = []
        for level_out in y_pred:
            offset = level_out[..., -4:]
            probabilities = tf.nn.sigmoid(level_out[..., :-4])
            out.append((offset, probabilities))
        return zip(*out)

    def _reshape_for_nms(self, tlbr, probabilities):
        for_nms = []
        batch_size = tf.shape(tlbr[0])[0]
        for box, prob in zip(tlbr, probabilities):
            box = tf.reshape(box, [batch_size, -1, 1, 4])
            prob = tf.reshape(prob, [batch_size, -1, self.num_classes])
            for_nms.append((box, prob))
        tlbr, prob = [tf.concat(x, axis=1) for x in zip(*for_nms)]
        return tlbr, prob

    def ground_truth_to_flat_tlbr_label(self, y_true):
        # todo this shouldnt really exist
        offset, label = PostProcessor._truth_unpack(y_true)
        tlbr_tensor = self.anchors.to_tlbr_tensor(offset)
        tlbr, probs = self._reshape_for_nms(
            tlbr_tensor,
            [tf.one_hot(x, depth=self.num_classes) for x in label])
        return self._apply_nms(tlbr, probs)

    @staticmethod
    def _truth_unpack(y_true):
        # todo this should really exist
        out = []
        for level_out in y_true:
            offset = level_out[..., -4:]
            label = tf.cast(level_out[..., :1], tf.int32)
            out.append((offset, label))
        return zip(*out)


class InferenceEfficientNet(tf.keras.models.Model):
    def __init__(self, efficient_det):
        super(InferenceEfficientNet, self).__init__()
        self.efficient_det = efficient_det
        self.post_processor = PostProcessor(efficient_det.anchors, self.efficient_det.num_classes)

    @tf.function
    def call(self, x, training=None, mask=None):
        model_out = self.efficient_det(x, training)
        boxes, score, label, valid_detections = self.post_processor.process_output(model_out)
        return boxes, score, label, valid_detections

    def process_ground_truth(self, y_true):
        return self.post_processor.ground_truth_to_flat_tlbr_label(y_true)
