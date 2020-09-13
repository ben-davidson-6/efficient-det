import tensorflow as tf
import efficient_det
from efficient_det.common.box import Boxes


class EfficientDetAnchors:

    def __init__(self, size, aspects, num_levels, iou_match_thresh=None):
        self.size = size
        self.aspects = tf.constant(aspects)
        self.num_levels = num_levels
        self.iou_match_thresh = iou_match_thresh
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
            # add fake batch
            if regression.ndim == 4:
                regression = tf.expand_dims(regression, axis=0)
            yield self._regress_to_absolute_individual_level(level, regression)

    def absolute_to_regression(self, boxes):
        """
        Take a  boxes object containing a tensor of bounding boxes
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
        boxes : efficient_det.common.box.Boxes
            contains tensor/array [nboxes, ymin, xmin, ymax, xmax]
            bounding boxes for a single image and integer
            tensor/array [nboxes] the class of each box

        Returns
        -------

        regressions:
            yield tensors of shape [h_i, w_i, n, 4] corresponding to the regressions
            from the default boxes as well as [h_i, w_i, n] indication the classes
            with nan being a negative
        """
        for level in range(self.num_levels):
            yield self._assign_boxes_to_level(boxes, level)

    def regressions_to_tlbr(self, regressions):
        """

        Parameters
        ----------
        regressions : generator of tuples (regression, regression_label) for a single image

        Returns
        -------

        box tensor : [n, 4] in tlbr format
        label tensor : [n]

        """
        tlbr_boxes = []
        labels = []
        for level, (regression_label, regression) in enumerate(regressions):
            boxes = self._regress_to_absolute_tlbr(level, regression)
            boxes_out = tf.boolean_mask(boxes, regression_label != efficient_det.NO_CLASS_LABEL)
            labels_out = tf.boolean_mask(regression_label, regression_label != efficient_det.NO_CLASS_LABEL)
            tlbr_boxes.append(boxes_out)
            labels.append(labels_out)
        return tf.concat(tlbr_boxes, axis=0), tf.concat(labels, axis=0)

    def _regress_to_absolute_individual_level(self, level, regression):
        default_boxes = self._default_boxes_for_regression(level, regression)
        offset, scale = tf.split(regression, num_or_size_splits=2, axis=-1)
        centroid, dimensions = tf.split(default_boxes, num_or_size_splits=2, axis=-1)
        regressed_centroids = centroid + offset
        regressed_dimensions = EfficientDetAnchors._scale_width_and_height(dimensions, scale)
        absolutes = tf.concat([regressed_centroids, regressed_dimensions], axis=-1)
        return absolutes

    def _regress_to_absolute_tlbr(self, level, regression):
        absolutes = self._regress_to_absolute_individual_level(level, regression)
        return Boxes.absolute_as_tlbr(absolutes)

    def _assign_boxes_to_level(self, boxes, level):
        """
        Given a level we can define the default boxes for that level.
        We can then mathc the defaults to the ground truth and so
        build the regressions

        Parameters
        ----------
        boxes : Boxes
        level : int


        Returns
        -------
        regressions : tensors of shape [1, h_i, w_i, n, 4] corresponding to the regressions
        classes : integer tensor of shape [h_i, w_i, n] indication the classes with nan indicating a negative
        """
        default_boxes = self._default_boxes_for_absolute(level, boxes)
        best_box_classes, best_ious, best_boxes = boxes.match_with_anchor(default_boxes)
        best_box_classes = tf.where(
            best_ious > self.iou_match_thresh,
            best_box_classes,
            tf.ones_like(best_box_classes)*efficient_det.NO_CLASS_LABEL)
        correspond_regression = EfficientDetAnchors._regression_from_boxes(default_boxes, best_boxes)
        return best_box_classes, correspond_regression

    @staticmethod
    def _regression_from_boxes(default_boxes, target_boxes):
        centroid_default, dimensions_default = tf.split(default_boxes, num_or_size_splits=2, axis=-1)
        centroid_target, dimensions_target = tf.split(target_boxes, num_or_size_splits=2, axis=-1)
        offset = centroid_target - centroid_default
        scale = tf.math.log(dimensions_target/dimensions_default)
        return tf.concat([offset, scale], axis=-1)

    @staticmethod
    def _scale_width_and_height(dimensions, scales):
        width_height_scale = tf.exp(scales)
        return dimensions*width_height_scale

    @staticmethod
    def _default_absolute_centroids(level, height, width):
        stride = EfficientDetAnchors._level_to_stride(level)
        pixel_coords = EfficientDetAnchors._get_pixel_coords(height, width)
        return pixel_coords*stride + stride//2

    def _default_width_height(self, level):
        stride = EfficientDetAnchors._level_to_stride(level)
        unit_aspect = stride*self.size
        return unit_aspect*self.aspects/2.

    def _default_box_tensor(self, level, height, width):
        # [h, w, 2] = [h, w, n, 2]
        default_centroids = EfficientDetAnchors._default_absolute_centroids(level, height, width)
        default_centroids = tf.tile(tf.expand_dims(default_centroids, axis=2), (1, 1, self.num_boxes(), 1))
        # [n, 2] -> [h, w, n, 2]
        width_height = self._default_width_height(level)
        width_height = tf.tile(width_height[None, None], (height, width, 1, 1))
        return tf.concat([default_centroids, width_height], axis=-1)

    @staticmethod
    def _level_to_stride(level):
        return 2**(3 + level)

    @staticmethod
    def _get_pixel_coords(height, width):
        xs = tf.range(width, dtype=tf.float32)
        ys = tf.range(height, dtype=tf.float32)
        return tf.stack(tf.meshgrid(xs, ys), axis=-1)

    def _default_boxes_for_regression(self, level, regression):
        h, w = EfficientDetAnchors._regression_height_width(regression)
        default_boxes = self._default_box_tensor(level, h, w)
        return default_boxes

    def _default_boxes_for_absolute(self, level, boxes):
        image_width, image_height = boxes.get_image_dimensions()
        stride = EfficientDetAnchors._level_to_stride(level)
        regress_height, regress_width = image_height//stride, image_width//stride
        return self._default_box_tensor(level, regress_height, regress_width)

    def num_boxes(self):
        return tf.shape(self.aspects)[0]

    @staticmethod
    def _regression_height_width(regression):
        shape = tf.shape(regression)
        if tf.rank(regression) == 5:
            return shape[1], shape[2]
        else:
            return shape[0], shape[1]