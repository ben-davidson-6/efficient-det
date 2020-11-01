import tensorflow as tf
import pathlib

from efficient_det.geometry.plot import draw_model_output
from efficient_det.model.model import InferenceEfficientNet
from efficient_det import NO_CLASS_LABEL


class TensorboardCallback(tf.keras.callbacks.Callback):
    MAX_EXAMPLES_PER_DATASET = 1
    VALIDATION_NAME = 'val'
    TRAINING_NAME = 'train'
    IMAGE_SIZE = 256

    def __init__(self, training_set: tf.data.Dataset, validation_set: tf.data.Dataset, logdir: str):
        super(TensorboardCallback, self).__init__()
        self.inference_net = None

        self.validation_examples = TensorboardCallback.extract_data_from_validation(validation_set)
        for image, regression in training_set.take(1):
            self.training_examples = (image, regression)
            break

        self.train_batch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        self.val_batch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        self.validation_gt_summary_images = None
        self.training_gt_summary_images = None

        self.validation_writer = TensorboardCallback._make_file_writer(logdir, TensorboardCallback.VALIDATION_NAME)
        self.training_writer = TensorboardCallback._make_file_writer(logdir, TensorboardCallback.TRAINING_NAME)

    def get_net(self):
        if self.inference_net is None:
            self.inference_net = InferenceEfficientNet(self.model)
        return self.inference_net

    def _draw_summary_images(self, with_model=False):
        thresh = 0.5 if with_model else 0.
        validation = self._get_validation_images(thresh, with_model)
        training = self._get_training_images(thresh, with_model)
        return validation, training

    def _draw_single_image(self, image, boxes, scores, thresh):
        for box, score in zip(boxes, scores):
            to_show = scores > thresh
            box = tf.boolean_mask(box, to_show)
            image = tf.image.draw_bounding_boxes(image, box, [(0., 0., 1.), (0., 1., 0.), (1., 0., 0.)])
        image = tf.image.resize(image, (TensorboardCallback.IMAGE_SIZE, TensorboardCallback.IMAGE_SIZE))
        return image

    def _get_validation_images(self, thresh, with_model):
        validation = []
        for image, offset in self.validation_examples:
            if with_model:
                box, label, score = self.get_net()(image, training=False)
            else:
                box, label = self.get_net().process_ground_truth(offset)
                score = tf.ones_like(label, dtype=tf.float32)
                box = tf.where(label[..., None] == NO_CLASS_LABEL, box, -1.)
            image = draw_model_output(image, box, score, thresh)
            image = tf.image.resize(image, (TensorboardCallback.IMAGE_SIZE, TensorboardCallback.IMAGE_SIZE))
            validation.append(image[0])
        return validation

    def _get_training_images(self, thresh, with_model):
        training = []
        for i in range(TensorboardCallback.MAX_EXAMPLES_PER_DATASET):
            image = self.training_examples[0][i][None]
            if with_model:
                box, label, score = self.get_net()(image, training=False)
            else:
                offset = [x[i:i+1] for x in self.training_examples[1]]
                box, label = self.get_net().process_ground_truth(offset)
                score = tf.ones_like(label, dtype=tf.float32)
                box = tf.where(label[..., None] == NO_CLASS_LABEL, box, -1.)

            image = draw_model_output(image, box, score, thresh)
            image = tf.image.resize(image, (TensorboardCallback.IMAGE_SIZE, TensorboardCallback.IMAGE_SIZE))
            training.append(image[0])
        return training

    def on_epoch_begin(self, epoch, logs=None):
        self._write_image_summaries(epoch)

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        with self.validation_writer.as_default():
            for key in logs:
                tf.summary.scalar(key, logs[key], step=self.val_batch_counter)
            self.val_batch_counter.assign_add(1)

    def on_train_batch_end(self, batch, logs=None):
        with self.training_writer.as_default():
            for key in [k for k in logs if not 'val_' in k]:
                tf.summary.scalar(key, logs[key], step=self.train_batch_counter)
        self.train_batch_counter.assign_add(1)

    def on_train_end(self, logs=None):
        # report final best stats
        pass

    def _write_image_summaries(self, epoch):
        validation, training = self._build_summary_image()
        val_hists, training_hists = self._build_histograms()
        with self.validation_writer.as_default():
            offsets, classes, scores, gt_labels = val_hists
            tf.summary.image('examples', validation, step=epoch)
            tf.summary.histogram('classes', classes, step=epoch)
            tf.summary.histogram('offsets', offsets, step=epoch)
            tf.summary.histogram('scores', scores, step=epoch)
            tf.summary.histogram('gt', gt_labels, step=epoch)
        with self.training_writer.as_default():
            offsets, classes, scores = training_hists
            tf.summary.image('examples', training, step=epoch)
            tf.summary.histogram('classes', classes, step=epoch)
            tf.summary.histogram('offsets', offsets, step=epoch)
            tf.summary.histogram('scores', scores, step=epoch)

    def write_metric_stats(self, logs):
        with self.validation_writer.as_default():
            for key in [k for k in logs if 'val_' in k]:
                tf.summary.scalar(key.replace('val_', ''), logs[key], step=self.train_batch_counter)

    @staticmethod
    def extract_data_from_validation(validation_set):
        data = []
        for image, offset in validation_set:
            data.append((image, offset))
            if len(data) == TensorboardCallback.MAX_EXAMPLES_PER_DATASET:
                break
        return data

    def _build_summary_image(self):
        if self.validation_gt_summary_images is None:
            validation, training = self._draw_summary_images(with_model=False)
            self.validation_gt_summary_images = tuple(validation)
            self.training_gt_summary_images = tuple(training)

        validation, training = self._draw_summary_images(with_model=True)
        validation_image = tf.concat(validation, axis=1)
        training_image = tf.concat(training, axis=1)
        validation_gt = tf.concat(self.validation_gt_summary_images, axis=1)
        training_gt = tf.concat(self.training_gt_summary_images, axis=1)
        validation = tf.concat([validation_gt, validation_image], axis=0)
        training = tf.concat([training_gt, training_image], axis=0)
        return validation[None], training[None]

    def _build_histograms(self):
        train_hists = []
        for i in range(TensorboardCallback.MAX_EXAMPLES_PER_DATASET):
            image = self.training_examples[0][i][None]
            box, label, score = self.get_net()(image, training=False)
            train_hists.append((box, label, score))

        val_hists = []
        for image, offset in self.validation_examples:
            box, label, score = self.get_net()(image, training=False)
            gt_labels = tf.concat([tf.reshape(o[..., 0], [-1]) for o in offset], axis=0)
            val_hists.append((box, label, score, gt_labels))

        val_hists = list(map(lambda x: tf.concat(x, axis=0), zip(*val_hists)))
        train_hists = list(map(lambda x: tf.concat(x, axis=0), zip(*train_hists)))
        return val_hists, train_hists

    @staticmethod
    def _make_file_writer(log_dir, name):
        writer_loc = pathlib.Path(log_dir).joinpath(name)
        return tf.summary.create_file_writer(str(writer_loc))