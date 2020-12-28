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

    def validation_set_for_final_eval(self, ):
        
        def mold_for_eval(x):
            bbox = x['faces']['bbox']
            image = tf.image.resize(x['image'], (1024, 1024))
            labels = tf.zeros_like(bbox, dtype=tf.int32)[:, 0]
            return (image, {'labels': labels, 'bboxes': bbox})

        ds = Faces.raw_dataset(Faces.validation).map(mold_for_eval)

        def gen():
            for image_id, (image, info) in enumerate(ds):
                info['image_id'] = image_id + 1
                yield (image, info)

        return tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.float32, {'labels': tf.int32, 'bboxes': tf.float32, 'image_id': tf.int32}))

    def categories(self):
        return [{'id': 1, 'name': 'face'}]
        
    @staticmethod
    def get_image_box_label(example):
        bbox = example['faces']['bbox']
        image = example['image']
        labels = tf.zeros_like(bbox, dtype=tf.int32)[:, 0]
        return image, bbox, labels

    @staticmethod
    def raw_dataset(split) -> tf.data.Dataset:
        shuffle = split == Faces.train
        def throw_away_invalid(x):
            bbox = x['faces']['bbox']
            width = bbox[:, 3] - bbox[:, 1]
            height = bbox[:, 2] - bbox[:, 0]
            ok = tf.logical_and(height > 0, width > 0)
            x['faces']['bbox'] = bbox[ok]
            return x
        return tfds.load(
            NAME,
            download=False,
            split=split,
            shuffle_files=shuffle).map(throw_away_invalid)


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
    import numpy as np
    boxes = []
    invalid = []
    for x, g in dataset.validation_set_for_final_eval():
        shape = np.array(x.shape[:2])
        shape = np.concatenate([shape, shape])[None]
        boxes.append(g['bboxes'].numpy()*shape)
        invalid.append(g['invalid'].numpy())
    invalid = np.concatenate(invalid, axis=0)
    print(np.sum(invalid))
    print(np.sum(~invalid))
    print(np.sum(invalid)/np.sum(~invalid))
    boxes = np.concatenate(boxes, axis=0)
    np.save('boxes', boxes)
