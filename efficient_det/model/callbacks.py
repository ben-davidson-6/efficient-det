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
    IMAGE_SIZE = 512
    FULL_EVAL_FREQ = 10

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
        if batch%100 == 0:
            with self.training_writer.as_default():
                tf.summary.scalar('loss', logs['loss'], step=self.train_batch_counter)
            self.train_batch_counter.assign_add(1)

    def on_test_batch_end(self, batch, logs=None):
        if batch%100 == 0:
            with self.validation_writer.as_default():
                tf.summary.scalar('loss', logs['loss'], step=self.val_batch_counter)
            self.val_batch_counter.assign_add(1)
    
    def on_epoch_end(self, epoch, logs=None):
        val_precision = self._averaged_metric(metric_name='precision', training=False, logs=logs)
        train_precision = self._averaged_metric(metric_name='precision', training=True, logs=logs)
        val_offset_err = self._averaged_metric(metric_name='offset_err', training=False, logs=logs)
        train_offset_err = self._averaged_metric(metric_name='offset_err', training=True, logs=logs)
        val_scale_err = self._averaged_metric(metric_name='scale_err', training=False, logs=logs)
        train_scale_err = self._averaged_metric(metric_name='scale_err', training=True, logs=logs)
        val_recall = self._averaged_metric(metric_name='recall', training=False, logs=logs)
        train_recall = self._averaged_metric(metric_name='recall', training=True, logs=logs)

        with self.validation_writer.as_default():
            tf.summary.scalar('metric/precision', val_precision, step=epoch)
            tf.summary.scalar('metric/offset_err', val_offset_err, step=epoch)
            tf.summary.scalar('metric/scale_err', val_scale_err, step=epoch)
            tf.summary.scalar('metric/recall', val_recall, step=epoch)

        with self.training_writer.as_default():
            tf.summary.scalar('metric/precision', train_precision, step=epoch)
            tf.summary.scalar('metric/offset_err', train_offset_err, step=epoch)
            tf.summary.scalar('metric/scale_err', train_scale_err, step=epoch)
            tf.summary.scalar('metric/recall', train_recall, step=epoch)

        if epoch%TensorboardCallback.FULL_EVAL_FREQ == 0 and epoch > 0:
            coco = self.coco_eval.evaluate_model(self.inference_net)
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

    def initialise_inference_net(self):
        if self.inference_net is None:
            self.inference_net = InferenceEfficientNet(self.model)

    def _draw_summary_images(self, with_model=False):
        validation = self._get_images(training=False, with_model=with_model)
        training = self._get_images(training=True, with_model=with_model)
        return validation, training

    def _averaged_metric(self, metric_name, training, logs):
        val = 'val'
        key_to_search = metric_name
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
        self.initialise_inference_net()
        for image, offset in examples:
            if with_model:
                # todo put the valid stuff inside the inference
                box, score, label, valid_detections = self.inference_net(image, training=False)
                valid_detections = valid_detections[0]
                box = box[:1, :valid_detections]
                score = score[:1, :valid_detections]
                label = label[:1, :valid_detections]
                thresh = 0.5
            else:
                box, score, label, _ = self.inference_net.process_ground_truth(offset)
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
        logdir_model = pathlib.Path(self.logdir).joinpath('model/weights')
        if ap > self.best_ap:
            self.best_ap = ap
            self.model.save_weights(logdir_model)

