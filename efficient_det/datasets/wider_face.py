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
            image, bbox, *_ = self.resize(x['image'], x['faces']['bbox'], None)
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
            download=True,
            split=split,
            shuffle_files=shuffle).map(throw_away_invalid)


if __name__ == '__main__':
    pass