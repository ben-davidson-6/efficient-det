import tensorflow as tf


class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            init_learning_rate,
            target_learn_after_1,
            num_steps_per_epoch,
            num_epochs):

        self.init_learning_rate = init_learning_rate
        self.target_learn_after_1 = target_learn_after_1
        self.num_steps_per_epoch = tf.cast(num_steps_per_epoch, tf.float32)
        self.num_epochs = num_epochs

        self.cosine_decay = tf.keras.experimental.CosineDecay(
            target_learn_after_1,
            decay_steps=num_steps_per_epoch*num_epochs)
        self.linear = tf.keras.optimizers.schedules.PolynomialDecay(
            init_learning_rate,
            num_steps_per_epoch,
            target_learn_after_1)

    def __call__(self, step):
        return tf.cond(
            step < self.num_steps_per_epoch,
            lambda: self.linear(step),
            lambda: self.cosine_decay(step - self.num_steps_per_epoch))         

    def get_config(self):
        return {
            'num_epochs': self.num_epochs,
            'num_steps_per_epoch': self.num_steps_per_epoch,
            'target_learn_after_1': self.target_learn_after_1,
            'init_learning_rate': self.init_learning_rate
        }
