import tensorflow as tf
import pathlib

from efficient_det.common import box, plot


class TensorboardCallback(tf.keras.callbacks.Callback):
    MAX_EXAMPLES_PER_DATASET = 4
    VALIDATION_NAME = 'val'
    TRAINING_NAME = 'train'
    IMAGE_SIZE = 256

    def __init__(self, training_set: tf.data.Dataset, validation_set: tf.data.Dataset, logdir: str):
        super(TensorboardCallback, self).__init__()
        # todo the training examples actually need to be processed a little
        #   we should have them in the same form as validation (image, regression)
        #   we should also put these into tlbr format so we can just draw them once?
        #   we can save the groundtruth and draw it once then at start of each epoch vstack them
        #   will need to reshape the val examples so they are same shape and we can display them all
        #   one colour for correct one for incorrect
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
        with self.validation_writer.as_default():
            tf.summary.image('examples', validation, step=epoch)
        with self.training_writer.as_default():
            tf.summary.image('examples', training, step=epoch)

    def write_metric_stats(self, logs):
        with self.validation_writer.as_default():
            for key in [k for k in logs if 'val_' in k]:
                tf.summary.scalar(key.replace('val_', ''), logs[key], step=self.train_batch_counter)

    def _draw_summary_images(self, with_model=False):
        validation = []
        thresh = 0. if with_model else 0.
        for image, regression in self.validation_examples:
            if with_model:
                regression = self.model(image, training=False)
            validation.append(self._draw_single_image(tf.cast(image, tf.uint8), regression, thresh))

        training = []
        for i in range(TensorboardCallback.MAX_EXAMPLES_PER_DATASET):
            image = self.training_examples[0][i][None]
            if with_model:
                regression = self.model(image, training=False)
            else:
                regression = [x[i] for x in self.training_examples[1]]
            training.append(self._draw_single_image(tf.cast(image, tf.uint8), regression, thresh))
        return validation, training

    @staticmethod
    def extract_data_from_validation(validation_set):
        data = []
        for image, regression in validation_set:
            data.append((image, regression))
            if len(data) == TensorboardCallback.MAX_EXAMPLES_PER_DATASET:
                break
        return data

    def _draw_single_image(self, image, regression, thresh):
        boxes, labels = self.model.anchors.model_out_to_tlbr(regression, thresh)
        bboxes = box.Boxes.from_image_boxes_labels(image[0], boxes, labels)
        plotter = plot.Plotter(image[0], bboxes, normalised=False)
        bboxes = plotter.draw_boxes()
        bboxes = tf.image.resize(bboxes, (TensorboardCallback.IMAGE_SIZE, TensorboardCallback.IMAGE_SIZE))
        return bboxes

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
        writer_loc = pathlib.Path(log_dir).joinpath(name)
        return tf.summary.create_file_writer(str(writer_loc))