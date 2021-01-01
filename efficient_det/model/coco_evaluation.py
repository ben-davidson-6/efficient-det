import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import json
import tempfile
import tensorflow as tf
import os

from efficient_det.geometry.box import TLBRBoxes
from efficient_det.constants import COCO_ANNOTATIONS
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pprint import pprint


class CocoEvaluation:
    def __init__(self, dataset, categories, coco_params={}, is_coco=False):
        self.dataset = dataset
        self.is_coco = is_coco
        if is_coco:
            self.coco_gt = COCO(annotation_file=COCO_ANNOTATIONS)
            self.label_to_category = tf.constant([x['id'] for x in self.coco_gt.dataset['categories']])
        else:
            self.coco_gt = None
            self.label_to_category = tf.range(1, len(categories) + 1, dtype=tf.int32)
        self.categories = categories
        self.coco_params = coco_params

    def evaluate_model(self, inference_model):
        coco_gt = self._get_coco_gt()
        results = self._get_results(inference_model)
        try:
            coco_res = coco_gt.loadRes(results)
        except IndexError:
            return -1.
        coco_eval = COCOeval(coco_gt, coco_res, 'bbox')
        coco_eval = self.update_params(coco_eval)
        coco_eval.params.imgIds = list(set([x['image_id'] for x in results]))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return CocoEvaluation._extract_metric(coco_eval)

    def _get_results(self, inference_model):
        results = []
        for image, gt in self.dataset:
            bboxes, scores, labels = CocoEvaluation._detect(
                inference_model, image)
            image_id = int(gt['image_id'].numpy())
            for ann in self._make_annotations(image_id, labels, bboxes, scores):
                results.append(ann)
        return results

    def _get_coco_gt(self):
        # {
        #   "images":[{"id": 73}],
        #   "annotations":[{"image_id":73,"category_id":1,"bbox":[10,10,50,100],"id":1,"iscrowd": 0,"area": 10}],
        #   "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}, {"id": 3, "name": "car"}]}

        if self.coco_gt is not None:
            return self.coco_gt

        image_ids = []
        annotations = []
        # must start at 1 for whatever reason
        k = 1
        for image, gt in self.dataset:
            bboxes = gt['bboxes'].numpy()
            labels = gt['labels'].numpy()
            image_id = int(gt['image_id'].numpy())
            image_ids.append({"id": image_id})

            image_h, image_w = image.shape[:2]
            bboxes = CocoEvaluation._get_coco_boxes(bboxes, image_h, image_w)
            for annotation in self._make_annotations(image_id, labels, bboxes.numpy()):
                annotation['id'] = k
                annotation['iscrowd'] = 0
                annotation['area'] = annotation['bbox'][2] * \
                    annotation['bbox'][3]
                k += 1
                annotations.append(annotation)
        coco_gt = {
            'images': image_ids,
            'annotations': annotations,
            'categories': self.categories
        }
        self.coco_gt = COCO()
        self.coco_gt.dataset = coco_gt
        self.coco_gt.createIndex()
        return self.coco_gt

    @staticmethod
    def _detect(inference_model, image):
        image_height, image_width = image.shape[:2]
        bbox, scores, labels, valid_detections = inference_model(
            image[None], training=False)
        bbox, scores, labels = CocoEvaluation._get_valid_detections(
            bbox, scores, labels, valid_detections)
        bbox = CocoEvaluation._get_coco_boxes(bbox, image_height, image_width)
        return bbox.numpy(), scores.numpy(), labels.numpy()

    def _make_annotations(self, image_id, labels, bboxes, scores=None):
        # {"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
        anns = zip(bboxes, scores, labels) if scores is not None else zip(
            bboxes, labels)
        for ann in anns:
            try:
                box, score, cat_id = ann
                cat_id = int(self.label_to_category[cat_id].numpy())
                yield {'image_id': image_id, 'category_id': cat_id, 'bbox': box.tolist(), 'score': score}
            except Exception as e:
                box, cat_id = ann
                cat_id = int(self.label_to_category[cat_id].numpy())
                yield {'image_id': image_id, 'category_id': cat_id, 'bbox': box.tolist()}

    @staticmethod
    def _get_coco_boxes(bbox, image_height, image_width):
        tlbr_box = TLBRBoxes(bbox)
        tlbr_box.unnormalise(image_height, image_width)
        return tlbr_box.as_coco_box_tensor()

    @staticmethod
    def _get_valid_detections(bbox, scores, labels, valid_detections):
        valid_detections = valid_detections[0]
        bbox = bbox[0, :valid_detections]
        scores = scores[0, :valid_detections]
        labels = labels[0, :valid_detections]
        return bbox, scores, labels

    @staticmethod
    def _extract_metric(coco_eval):
        # todo how to set this up properly
        p = coco_eval.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == 'all']
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == 100]
        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval['precision']
        # IoU
        s = s[:, :, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def update_params(self, coco_eval):
        for k in self.coco_params:
            coco_eval.params.__setattr__(k, self.coco_params[k])
        return coco_eval


if __name__ == '__main__':
    import time
    import efficient_det.model as model
    import efficient_det.datasets.wider_face as wider_face
    import efficient_det.geometry.plot as p
    import efficient_det.datasets.train_data_prep as prep
    import matplotlib.pyplot as plt

    anchor_size = 2.
    base_aspects = [
        (1., 1.),
        (0.75, 1.5),
        (1.5, 0.75)
    ]
    aspects = []
    n_octave = 3
    for octave in range(n_octave):
        scale = 2**(octave / n_octave)
        for aspect in base_aspects:
            aspects.append((aspect[0] * scale, aspect[1] * scale))
    num_levels = 6
    anchors = model.build_anchors(
        anchor_size, num_levels=num_levels, aspects=aspects)

    

    prepper = prep.ImageBasicPreparation(
        min_scale=0.5,
        overlap_percentage=0.3,
        max_scale=1.2,
        target_shape=512)
    ds = wider_face.Faces(anchors, lambda x,y,z: (x,y,z), prepper, 0.5, 0.4, 1)
    # network
    phi = 0
    num_classes = 1
    efficient_det = model.EfficientDetNetwork(
        phi, num_classes, anchors, n_extra_downsamples=3)
    # efficient_det.load_weights(
    #     'C:\\Users\\bne\\PycharmProjects\\efficient-det\\artifacts\\Dec_28_150139\\model\\model')
    inference_net = model.InferenceEfficientNet(efficient_det)
    coco = CocoEvaluation(
        ds.validation_set_for_final_eval(),
        ds.categories(),
        coco_params={'iouThrs': np.array([0.5])})
    # coco.evaluate_model(inference_net)
    for x, y in ds.validation_set():
        box, score, label, valid_detections = inference_net.process_ground_truth(y)
        valid_detections = valid_detections[0]
        box = box[:1, :valid_detections]
        score = score[:1, :valid_detections]
        label = label[:1, :valid_detections]
        im = p.draw_model_output(x, box, score, thresh=0.)
        plt.imshow(im[0])
        plt.show()
