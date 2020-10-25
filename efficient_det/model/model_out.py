import tensorflow as tf


class ModelOutput:
    def __init__(self, regressions):
        self.regression = [SingleLevelOut(x, level) for level, x in enumerate(regressions)]

    def to_detection(self):
        outputs = []
        for reg in self.regression:
            outputs.append(reg.to_detection())
        boxes, scores, labels = map(lambda x: tf.concat(x, axis=0), zip(*outputs))
        return boxes, scores, labels


class SingleLevelOut:
    def __init__(self, model_out_single, level):
        self.model_output_single = model_out_single
        self.level = level
        self.height, self.width = SingleLevelOut._output_dimensions(model_out_single)

    def to_detection(self,):
        offset, label, score = self._unpack()

    def _unpack(self):
        offset = tf.reshape(self.model_output_single[..., -4:], [-1, 4])
        probabilities = tf.reshape(tf.nn.sigmoid(self.model_output_single[..., :-4]), [-1, 1])
        score = tf.reshape(tf.reduce_max(probabilities, axis=-1, keepdims=True), [-1, 1])
        label = tf.reshape(tf.argmax(score, axis=-1, keepdims=True), [-1, 1])
        return offset, label, score

    @staticmethod
    def _output_dimensions(regression):
        shape = tf.shape(regression)
        return shape[1], shape[2]
