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
    def validation_set_for_final_eval():
        def mold_for_final(x):
            image = x['image']
            image = tf.image.resize(image, (512, 512))
            bbox = x['objects']['bbox']
            labels = x['objects']['label']
            return image, {'bboxes': bbox, 'labels': labels, 'image_id': x['image/id']}
        return Coco.raw_dataset(Coco.validation).map(mold_for_final)

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

    def categories(self):
        pass

if __name__ == '__main__':

    import efficient_det.model as model
    import efficient_det.datasets.train_data_prep as train_data_prep
    # anchors
    anchor_size = 3
    aspects = [
        (1., 1.),
        (.75, 1.5),
        (1.5, 0.75),
    ]
    anchors = model.build_anchors(anchor_size, num_levels=6, aspects=aspects)
    net = model.EfficientDetNetwork(0, 80, anchors)
    # dataset
    prepper = train_data_prep.ImageBasicPreparation(min_scale=0.8, max_scale=1.2, target_shape=512)
    iou_match_thresh = 0.3
    dataset = Coco(
        anchors=anchors,
        augmentations=None,
        basic_training_prep=prepper,
        iou_thresh=iou_match_thresh,
        batch_size=1)
