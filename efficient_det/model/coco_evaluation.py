import pprint
import tensorflow as tf
import tempfile
import json
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from efficient_det.geometry.box import TLBRBoxes
from efficient_det.datasets.coco import Coco


coco_gt = COCO(annotation_file='C:\\Users\\bne\\tensorflow_datasets\\annotations\\instances_val2017.json')
label_to_category_id = tf.constant([x['id'] for x in coco_gt.dataset['categories']])


def evaluate_coco_model(inference_model):
    get_validation_result(Coco.validation_set_for_final_eval(), inference_model)
    # testing_result(coco.test_set(), inference_model)


def evaluate_results(results):
    coco_res = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_res, 'bbox')
    coco_eval.params.imgIds = list(set([x['image_id'] for x in results]))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def get_validation_result(ds, inference_model):
    validation_results = []
    for example in ds:
        validation_results += _detect_as_coco_format(inference_model, example)
    evaluate_results(validation_results)


def _detect_as_coco_format(inference_model, example):
    # {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
    image_id, image = _parse_example(example)
    bbox, scores, category_ids = _detect(inference_model, image)
    return [
        {'image_id': image_id.numpy(), 'category_id': cat_id, 'bbox': box.tolist(), 'score': score}
        for box, score, cat_id in zip(bbox, scores, category_ids)]


def _detect(inference_model, image):
    image_height, image_width = image.shape[:2]
    image = tf.image.resize(image, (512, 512))
    bbox, scores, labels, valid_detections = inference_model(image[None], training=False)
    bbox, scores, labels = _get_valid_detections(bbox, scores, labels, valid_detections)
    bbox = _get_coco_boxes(bbox, image_height, image_width)
    category_ids = _get_coco_categories(labels)
    return bbox.numpy(), scores.numpy(), category_ids.numpy()


def _parse_example(example):
    image_id = example['image/id']
    image = example['image']
    return image_id, image


def _get_coco_boxes(bbox, image_height, image_width):
    tlbr_box = TLBRBoxes(bbox)
    tlbr_box.unnormalise(image_height, image_width)
    return tlbr_box.as_coco_box_tensor()


def _get_coco_categories(labels):
    return tf.gather(label_to_category_id, labels)


def _get_valid_detections(bbox, scores, labels, valid_detections):
    valid_detections = valid_detections[0]
    bbox = bbox[0, :valid_detections]
    scores = scores[0, :valid_detections]
    labels = labels[0, :valid_detections]
    return bbox, scores, labels


if __name__ == '__main__':
    import efficient_det.model as model

    # anchors
    anchor_size = 4
    base_aspects = [
        (1., 1.),
        (.75, 1.5),
        (1.5, 0.75),
    ]
    aspects = []
    for octave in range(3):
        scale = 2 ** (octave / 3)
        for aspect in base_aspects:
            aspects.append((aspect[0] * scale, aspect[1] * scale))
    num_levels = 5
    anchors = model.build_anchors(anchor_size, num_levels=num_levels, aspects=aspects)

    # network
    phi = 0
    num_classes = 80
    efficient_det = model.EfficientDetNetwork(phi, num_classes, anchors, n_extra_downsamples=2)
    efficient_det.load_weights('C:\\Users\\bne\\PycharmProjects\\efficient-det\\artifacts\\models\\Nov_12_192003\\model')
    inference_net = model.InferenceEfficientNet(efficient_det)
    evaluate_coco_model(inference_net)