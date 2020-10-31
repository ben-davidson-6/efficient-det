import os
# print('delet me')
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64'
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import math

from efficient_det.model.components.backbone import Backbone
from efficient_det.model.components.downsampler import Downsampler
from efficient_det.model.components.bifpn import BiFPN
from efficient_det.model.components.detection_head import DetectionHead


class EfficientDetNetwork(tf.keras.Model):

    def __init__(self, phi, num_classes, anchors, n_extra_downsamples=3):
        # todo could validate num levels here
        super(EfficientDetNetwork, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors.aspects)
        self.phi = phi
        self.backbone = self.get_backbone()
        self.downsampler = Downsampler(depth=64, n_extra=n_extra_downsamples)
        self.bifpn = self.get_bifpn()
        self.detection_head = self.get_detection_head()

    @tf.function
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


class PostProcessor:
    def __init__(self, anchors):
        self.anchors = anchors

    def process_output(self, model_out):
        flat_tlbr, flat_label, flat_score = self.model_out_to_flat_tlbr_label_score(model_out)
        # unique, idx = tf.unique(flat_label)
        # print(flat_score)
        # box_indices = tf.image.non_max_suppression(flat_tlbr, flat_score, 50)
        # tlbr = tf.gather(flat_tlbr, box_indices)
        # label = tf.gather(flat_label, box_indices)
        # score = tf.gather(flat_score, box_indices)
        return flat_tlbr, flat_label, flat_score

    def model_out_to_flat_tlbr_label_score(self, y_pred):
        offset_tensors, label, score = PostProcessor._predicted_unpack(y_pred)
        tlbr_tensor = self.anchors.to_tlbr_tensor(offset_tensors)
        flat_tlbr, flat_label, flat_score = PostProcessor._flatten_tensors_and_list(
            tlbr_tensor,
            label,
            score)
        return flat_tlbr, flat_label, flat_score

    def ground_truth_to_flat_tlbr_label(self, y_true):
        offset, label = PostProcessor._truth_unpack(y_true)
        tlbr_tensor = self.anchors.to_tlbr_tensor(offset)
        flat_tlbr, flat_label, _ = PostProcessor._flatten_tensors_and_list(
            tlbr_tensor,
            label,
            [tf.ones_like(x) for x in label])
        return flat_tlbr, flat_label

    @staticmethod
    def _predicted_unpack(y_pred):
        out = []
        for level_out in y_pred:
            offset = level_out[..., -4:]
            probabilities = tf.nn.sigmoid(level_out[..., :-4])
            score = tf.reduce_max(probabilities, axis=-1, keepdims=True)
            label = tf.argmax(score, axis=-1)[..., None]
            out.append((offset, label, score))
        return zip(*out)

    @staticmethod
    def _truth_unpack(y_true):
        out = []
        for level_out in y_true:
            offset = level_out[..., -4:]
            label = tf.cast(level_out[..., :1], tf.int32)
            out.append((offset, label))
        return zip(*out)

    @staticmethod
    def _flatten_tensors_and_list(tlbr, label, score):
        flattened = []
        batch_size = tf.shape(tlbr[0])[0]
        for box, l, s in zip(tlbr, label, score):
            box = tf.reshape(box, [batch_size, -1, 4])
            l = tf.reshape(l, [batch_size, -1])
            s = tf.reshape(s, [batch_size, -1])
            flattened.append((box, l, s))
        tlbr, l, s = [tf.concat(x, axis=1) for x in zip(*flattened)]
        return tlbr, l, s


class InferenceEfficientNet:
    def __init__(self, efficient_det):
        self.efficient_det = efficient_det
        self.post_processor = PostProcessor(efficient_det.anchors)

    def __call__(self, x, training):
        model_out = self.efficient_det(x, training)
        boxes, label, score = self.post_processor.process_output(model_out)
        return boxes, label, score

    def process_output(self, model_out):
        return self.post_processor.process_output(model_out)


if __name__ == '__main__':
    import efficient_det.model as model
    import efficient_det.datasets.coco as coco
    import matplotlib.pyplot as plt
    from efficient_det.geometry.plot import draw_model_output


    # anchors
    anchor_size = 4
    aspects = [
        (1., 1.),
        (.75, 1.5),
        (1.5, 0.75),
    ]
    anchors = model.build_anchors(anchor_size, num_levels=6, aspects=aspects)

    dataset = coco.Coco(
        anchors=anchors,
        augmentations=None,
        basic_training_prep=None,
        iou_thresh=0.1,
        batch_size=4)

    # network
    phi = 0
    num_classes = 80
    efficient_det = model.EfficientDetNetwork(phi, num_classes, anchors)
    efficient_det.load_weights('C:\\Users\\bne\\PycharmProjects\\efficient-det\\artifacts\\models\\Oct_31_102050\\model')

    inference_model = InferenceEfficientNet(efficient_det)
    for x, y in dataset.validation_set():
        box, label, score = inference_model(x, training=False)
        image = draw_model_output(x, box, score, 0.5)
        plt.imshow(image[0])
        plt.show()