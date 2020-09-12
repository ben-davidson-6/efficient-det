import tensorflow_datasets as tfds
import efficient_det.model.anchor
import tensorflow as tf
import efficient_det.datasets.augs as augs

NAME = 'coco/2017'


# todo need to cutout boxes and things like that
#   when resizing the image


class Coco:
    validation = 'validation'
    train = 'train'

    def __init__(self, batch_size, anchors, image_size):
        self.batch_size = batch_size
        self.anchors = anchors
        self.image_size = image_size

    def validation_set(self):
        ds = Coco.raw_dataset(Coco.validation)
        ds = ds.map(self.parse_single_example)
        ds = ds.map(self.build_regressions)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds

    @staticmethod
    def get_image_and_label(example):
        image = example['image']
        bbox = example['objects']['bbox']
        labels = example['objects']['label']
        return image, bbox, labels

    def parse_single_example(self, example):
        image, bboxes, labels = Coco.get_image_and_label(example)
        return image, bboxes, labels

    def build_regressions(self, image, bboxes, labels):
        """bbox should be normalised coords"""
        image_shape = tf.shape(image)
        height, width = image_shape[0], image_shape[1]
        gt_boxes = efficient_det.model.anchor.Boxes(
            height,
            width,
            bboxes,
            labels)
        gt_boxes.unnormalise()
        labels, regressions = self.anchors.absolute_to_regression(gt_boxes)
        regressions = regressions[0]
        return image, regressions, labels

    @staticmethod
    def raw_dataset(split) -> tf.data.Dataset:
        shuffle = split == Coco.train
        return tfds.load(
            NAME,
            download=False,
            split=split,
            shuffle_files=shuffle)


