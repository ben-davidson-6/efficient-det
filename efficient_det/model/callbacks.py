import tensorflow as tf
import pathlib
import numpy as np

from efficient_det.geometry.plot import draw_model_output
from efficient_det.model.model import InferenceEfficientNet
from efficient_det.model.coco_evaluation import CocoEvaluation


class TensorboardCallback(tf.keras.callbacks.Callback):
    MAX_EXAMPLES_PER_DATASET = 4
    VALIDATION_NAME = 'val'
    TRAINING_NAME = 'train'
    IMAGE_SIZE = 256
    FULL_EVAL_FREQ = 1

    def __init__(self, dataset, logdir: str, coco_parmas: dict = {}, is_coco: bool = False, draw_first: bool = True):
        super(TensorboardCallback, self).__init__()
        self.inference_net = None
        self.draw_first = draw_first
        self.logdir = logdir
        self.best_ap = -2.0
        self.coco_eval = CocoEvaluation(
            dataset.validation_set_for_final_eval(),
            dataset.categories(),
            is_coco=is_coco,
            coco_params=coco_parmas)
        self.validation_examples = TensorboardCallback.extract_images(dataset.validation_set())
        self.training_examples = TensorboardCallback.extract_images(dataset.training_set())

        self.train_batch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        self.val_batch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        self.validation_gt_summary_images = None
        self.training_gt_summary_images = None

        self.validation_writer = TensorboardCallback._make_file_writer(logdir, TensorboardCallback.VALIDATION_NAME)
        self.training_writer = TensorboardCallback._make_file_writer(logdir, TensorboardCallback.TRAINING_NAME)

    def on_epoch_begin(self, epoch, logs=None):
        self._write_image_summaries(epoch)

    def on_train_batch_end(self, batch, logs=None):
        if batch%100 != 0:
            return
        with self.training_writer.as_default():
            tf.summary.scalar('loss', logs['loss'], step=self.train_batch_counter)
        self.train_batch_counter.assign_add(1)

    def on_test_batch_end(self, batch, logs=None):
        pass
        # if batch%100 != 0:
        #     return
        # with self.validation_writer.as_default():
        #     tf.summary.scalar('loss', logs['loss'], step=self.val_batch_counter)
        # self.val_batch_counter.assign_add(1)
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch%TensorboardCallback.FULL_EVAL_FREQ == 0:
            coco = self.coco_eval.evaluate_model(self.get_net())
            with self.validation_writer.as_default():
                tf.summary.scalar('coco_eval', coco, step=epoch)
            self._save_weights(coco)

    def _write_image_summaries(self, epoch):
        if epoch == 0 and not self.draw_first:
            return
        validation, training = self._build_summary_image()
        with self.validation_writer.as_default():
            tf.summary.image('examples', validation, step=epoch)
        with self.training_writer.as_default():
            tf.summary.image('examples', training, step=epoch)

    def get_net(self):
        if self.inference_net is None:
            self.inference_net = InferenceEfficientNet(self.model)
        return self.inference_net

    def _draw_summary_images(self, with_model=False):
        validation = self._get_images(training=False, with_model=with_model)
        training = self._get_images(training=True, with_model=with_model)
        return validation, training

    def _average_precision(self, training, logs):
        val = 'val'
        key_to_search = 'precision'
        vals = []
        for k in [x for x in logs if key_to_search in x]:
            if training and val not in k:
                vals.append(logs[k])
            elif not training and val in k:
                vals.append(logs[k])
        return tf.reduce_mean(vals)

    def _get_images(self, training, with_model):
        images = []
        examples = self._get_examples(training)
        for image, offset in examples:
            if with_model:
                box, score, label, _ = self.get_net()(image, training=False)
                thresh = 0.5
            else:
                box, score, label, _ = self.get_net().process_ground_truth(offset)
                thresh = 0.0
            image = draw_model_output(image, box, score, thresh)
            image = tf.image.resize(image, (TensorboardCallback.IMAGE_SIZE, TensorboardCallback.IMAGE_SIZE))
            images.append(image[0])
        return images

    def _get_examples(self, train):
        if train:
            dataset = self.training_examples
        else:
            dataset = self.validation_examples
        return dataset

    @staticmethod
    def extract_images(dataset):
        return [
            (image[:1], [o[:1] for o in offset]) 
            for image, offset in dataset.take(TensorboardCallback.MAX_EXAMPLES_PER_DATASET)]

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

    @staticmethod
    def _make_file_writer(log_dir, name):
        writer_loc = pathlib.Path(log_dir).joinpath(f'logs/{name}')
        return tf.summary.create_file_writer(str(writer_loc))

    def _save_weights(self, ap):
        logdir = pathlib.Path(self.logdir).joinpath(f'model/model')
        if ap > self.best_ap:
            self.best_ap = ap
            self.model.save_weights(logdir)

