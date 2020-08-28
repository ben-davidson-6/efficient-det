import tensorflow as tf


class BBoxAdapter:
    def __init__(self, bboxes, image_sizes):
        self.bboxes = bboxes
        self.image_sizes = image_sizes

    def as_centroids_heights_widths(self):
        raise NotImplemented


class EfficientDetAnchors:

    def __init__(self, size, aspects):
        self.size = size
        self.aspects = tf.constant(aspects)
        tf.debugging.assert_rank(aspects, 2)

    def regression_to_absolute(self, regressions):
        """
        Convert a list of regressions into a list of absolute box coordinates, in
        unnormalised pixel coordinates

        Parameters
        ----------
        regressions : list of arrays of shape [batch_size, h_i, w_i, n, 4]
            each box must be written (ox, oy, sx, sy) where exp(sx) and exp(sy)
            scale default height and width, and ox, oy offset default centroids
            with default centroids being defined by the leve/stride implicit in the
            list.

        Returns
        -------
        output : yield array of shape [batch_size, h_i, w_i, n, 4]
            (cx, cy, w/2, h/2) center x, y and widht and height in pixels
        """
        for level, regression in enumerate(regressions):
            yield self._regress_to_absolute_individual_level(level, regression)

    def absolute_to_regression(self, absolute, classes):
        """
        Take a tensor of bounding boxes in the form [nboxes, ymin, xmin, ymax, xmax]
        and their corresponding classes [nboxes] return a list of tensors like
        [h, w, class, nanchors, 4] with the required offsets, where the class may
        for each level be nan indicating that no object sufficiently overlaps with
        a default box there.

        Note that this entails a choice, as two boxes could sufficiently overlap, in
        such a case we take the box with the best overlap or randomly if equal

        Note here we need to know how many levels the model has, we should probably read this
        from the model.

        the idea is to run this offline, transforming the dataset ahead of time, so that
        in the training loop we can just read the tensors
        Parameters
        ----------
        absolute

        Returns
        -------

        """
        raise NotImplemented

    def _regress_to_absolute_individual_level(self, level, regression):
        centroids = EfficientDetAnchors._regressed_centroids(level, regression)
        width_heights = self._regressed_box_shapes(level, regression)
        return tf.stack([centroids, width_heights], axis=-1)

    @staticmethod
    def _regressed_centroids(level, regression):
        centroid_offset_xy = regression[..., :2]
        default_centroids = EfficientDetAnchors._default_absolute_centroids(level, centroid_offset_xy)
        return default_centroids + centroid_offset_xy

    def _regressed_box_shapes(self, level, regression):
        width_height_scale = EfficientDetAnchors._get_width_height_scale(regression)
        default_width_height = self._default_width_height(level)
        return default_width_height*width_height_scale/2.

    @staticmethod
    def _default_absolute_centroids(level, tensor):
        pixel_coords = EfficientDetAnchors._get_pixel_coords(tensor)
        # add batch , and number of anchors dimensions
        pixel_coords = tf.expand_dims(pixel_coords, 0)
        pixel_coords = tf.expand_dims(pixel_coords, 3)
        return pixel_coords*EfficientDetAnchors._level_to_stride(level)

    def _default_width_height(self, level):
        stride = EfficientDetAnchors._level_to_stride(level)
        unit_aspect = stride*self.size
        # add batch height and width dimensions
        aspects_for_mult = self.aspects[None, None, None]
        return unit_aspect*aspects_for_mult

    @staticmethod
    def _level_to_stride(level):
        return 2**(3 + level)

    @staticmethod
    def _get_pixel_coords(tensor):
        tf.debugging.assert_rank(tensor, 5)
        tensor_shape = tf.shape(tensor)
        height = tensor_shape[1]
        width = tensor_shape[2]
        xs = tf.range(width, dtype=tf.float32)
        ys = tf.range(height, dtype=tf.float32)
        return tf.stack(tf.meshgrid(xs, ys), axis=-1)

    @staticmethod
    def _get_width_height_scale(regression):
        log_width_height = regression[..., 2:]
        width_height = tf.exp(log_width_height)
        return width_height


# class Anchor:
#     def __init__(self, stride, scale, octave, aspect):
#         self.scale = scale
#         self.stride = stride
#         self.octave = octave
#         self.aspect = aspect
#
#     def as_rcnn_box(self):
#         pass
#     def height_and_width(self):
#         return self._height(), self._width()
#
#     def _dimension_size(self, dimension):
#         return self.stride[dimension] * self.aspect[dimension] * 2**self.octave * self.scale
#
#     def _height(self, ):
#         return self._dimension_size(0)
#
#     def _width(self):
#         return self._dimension_size(1)
#
#
# class AnchorsPerLevel:
#     def __init__(self, levels):


# anchor_config = [{size, aspect}]
# model = Model(anchor_config)
# model(x) = [(classification, regressions)]
# model.to_absolute(regressions)

# dataset = Dataset(model)