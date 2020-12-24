import tensorflow as tf


class Backbone(tf.keras.layers.Layer):

    def __init__(self, model_name):
        super(Backbone, self).__init__(name=model_name)
        self.backbone = Backbone.application_factory(model_name)

    def call(self, x, training=None):
        return self.backbone(x, training=False)

    @staticmethod
    def application_factory(phi):
        application_builder, layer_indices = Backbone.application_selector(phi)
        backbone = application_builder(include_top=False, input_shape=[None, None, 3])
        backbone = Backbone._freeze_some_layers(backbone, layer_indices[1])
        backbone_feature_tensors = [backbone.layers[index].output for index in layer_indices]
        model = tf.keras.Model(backbone.input, outputs=backbone_feature_tensors)
        return model

    @staticmethod
    def _freeze_some_layers(model, layer_index,):
        weights = model.get_weights()
        for k, layer in enumerate(model.layers):
            if type(model.get_layer(layer.name)) == tf.keras.layers.BatchNormalization:
                model.get_layer(layer.name).trainable = False
            if k < layer_index:
                model.get_layer(layer.name).trainable = False
        model = tf.keras.models.model_from_json(model.to_json())
        model.set_weights(weights)
        return model

    @staticmethod
    def application_selector(phi):
        if phi == 0:
            return tf.keras.applications.EfficientNetB0, (77, 164, 236)
        elif phi == 1:
            return tf.keras.applications.EfficientNetB1, (119, 236, 338)
        elif phi == 2:
            return tf.keras.applications.EfficientNetB2, (119, 236, 338)
        elif phi == 3:
            return tf.keras.applications.EfficientNetB3, (119, 266, 383)
        elif phi == 4:
            return tf.keras.applications.EfficientNetB4, (149, 326, 473)
        elif phi == 5:
            return tf.keras.applications.EfficientNetB5, (191, 398, 575)
        elif phi == 6:
            return tf.keras.applications.EfficientNetB6, (221, 458, 665)
        elif phi == 7:
            return tf.keras.applications.EfficientNetB7, (263, 560, 812)


if __name__ == '__main__':
    model = Backbone(0)