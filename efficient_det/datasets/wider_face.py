import tensorflow_datasets as tfds
import efficient_det.datasets.dataset
import efficient_det.model.anchor
import tensorflow as tf

NAME = 'wider_face'


class Faces(efficient_det.datasets.dataset.Dataset):

    def _raw_training_set(self,):
        ds = Faces.raw_dataset(Faces.train)
        ds = ds.map(Faces.get_image_box_label)
        return ds

    def _raw_validation_set(self,):
        ds = Faces.raw_dataset(Faces.validation)
        ds = ds.map(Faces.get_image_box_label)
        return ds

    @staticmethod
    def validation_set_for_final_eval():
        return Faces.raw_dataset(Faces.validation)

    @staticmethod
    def get_image_box_label(example):
        bbox = example['faces']['bbox']
        image = example['image']
        labels = tf.zeros_like(bbox, dtype=tf.int32)[:, 0]
        return image, bbox, labels

    @staticmethod
    def raw_dataset(split) -> tf.data.Dataset:
        shuffle = split == Faces.train
        return tfds.load(
            NAME,
            download=False,
            split=split,
            shuffle_files=shuffle)


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
    prepper = train_data_prep.ImageBasicPreparation(overlap_percentage=0.3, min_scale=0.8, max_scale=1.2, target_shape=512)
    iou_match_thresh = 0.3
    dataset = Faces(
        anchors=anchors,
        augmentations=None,
        basic_training_prep=prepper,
        iou_thresh=iou_match_thresh,
        batch_size=1)

    k = 1
    for x in dataset._raw_training_set():
        print(x)
        k += 1
        if k == 5:

            break
