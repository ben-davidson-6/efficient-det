import tensorflow_datasets as tfds
import efficient_det.datasets.dataset
import efficient_det.model.anchor
import tensorflow as tf

NAME = 'coco/2017'


class Coco(efficient_det.datasets.dataset.Dataset):

    def _raw_training_set(self,):
        ds = Coco.raw_dataset(Coco.train)
        ds = ds.map(Coco.get_image_box_label)
        return ds

    def _raw_validation_set(self,):
        ds = Coco.raw_dataset(Coco.validation)
        ds = ds.map(Coco.get_image_box_label)
        return ds

    @staticmethod
    def get_image_box_label(example):
        image = example['image']
        bbox = example['objects']['bbox']
        labels = example['objects']['label']
        return image, bbox, labels

    @staticmethod
    def raw_dataset(split) -> tf.data.Dataset:
        shuffle = split == Coco.train
        return tfds.load(
            NAME,
            download=False,
            split=split,
            shuffle_files=shuffle)


