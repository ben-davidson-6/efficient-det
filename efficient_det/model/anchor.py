import tensorflow as tf
import efficient_det

from efficient_det.geometry.box import Boxes, DefaultAnchorBoxes, TLBRBoxes, DefaultAnchorOffsets, CentroidWidthBoxes
from efficient_det.geometry.common import level_to_stride, pixel_coordinates


class SingleAnchorConfig:
    def __init__(self, size, aspect):
        self.size = size
        self.aspect = aspect

    def box_size(self, level):
        stride = level_to_stride(level)
        unit_aspect = stride * self.size
        return unit_aspect*self.aspect


class AnchorsAtLevel:
    def __init__(self, level, configs):
        self.level = level
        self.anchor_builders = [SingleAnchorAtLevelBuilder(config, level) for level, config in enumerate(configs)]

    def match_boxes(self, boxes: TLBRBoxes, image_height: int, image_width: int):
        anchors = self._build_anchors(image_height, image_width)
        ious = anchors.iou(boxes, as_original_shape=True)
        max_iou = tf.reduce_max(ious, axis=-1)
        best_box = tf.argmax(ious, axis=-1)
        best_box = boxes.create_box_tensor_from_indices(best_box)
        return TLBRBoxes(best_box), max_iou

    def to_tlbr_tensor(self, default_box_offsets: tf.Tensor):
        """regression should be height, width, n_anchors, 4"""
        image_height, image_width = self._original_image_size_from_default_offsets(default_box_offsets)
        anchors = self._build_anchors(image_height, image_width)
        offsets = DefaultAnchorOffsets(default_box_offsets)
        tlbr_boxes = offsets.as_centroid_width_box(anchors, as_original_shape=True)
        return tlbr_boxes

    def to_offset_tensor(self, boxes: TLBRBoxes, image_height: int, image_width: int):
        best_box, ious = self.match_boxes(boxes, image_height, image_width)
        anchors = self._build_anchors(image_height, image_width)
        centroid_boxes = best_box.as_centroid_and_width_box()
        offset = centroid_boxes.as_offset_boxes(anchors, as_original_shape=True)
        return offset, ious

    def _build_anchors(self, image_height, image_width):
        anchor_tensors = tf.stack([a.build_anchors(image_height, image_width).get_tensor() for a in self.anchor_builders], axis=-1)
        return CentroidWidthBoxes(anchor_tensors).as_tlbr_box()

    def _original_image_size_from_default_offsets(self, regression):
        regression_shape = tf.shape(regression)
        image_height = level_to_stride(self.level)*regression_shape[0]
        image_width = level_to_stride(self.level)*regression_shape[1]
        return image_height, image_width


class SingleAnchorAtLevelBuilder:

    def __init__(self, single_anchor_config, level):
        self.config = single_anchor_config
        self.level = level

    def build_anchors(self, image_height, image_width):
        stride_height, stride_width = self._normalised_stride(image_height, image_width)
        grid_height, grid_width = self._grid_dimensions(image_height, image_width)
        box_height, box_width = self._box_shape(image_height, image_width)
        anchors = DefaultAnchorBoxes(
            stride_height,
            stride_width,
            grid_height,
            grid_width,
            box_height,
            box_width)
        return anchors

    def _normalised_stride(self, image_height, image_width):
        stride = level_to_stride(self.level)
        return stride/image_height, stride/image_width

    def _grid_dimensions(self, image_height, image_width):
        stride = level_to_stride(self.level)
        return image_height//stride, image_width//stride

    def _box_shape(self, image_height, image_width):
        box_shape = self.config.box_size(self.level)
        normalised_box_shape = box_shape / tf.constant([image_height, image_width])
        return normalised_box_shape[0], normalised_box_shape[1]


class Anchors:
    def __init__(self, num_levels, configs):
        self.num_levels = num_levels
        self.configs = configs
        self.anchors = self._build_anchors_at_level()

    def to_tlbr_tensor(self, default_box_offsets: tf.Tensor):
        return [a.to_tlbr_tensor(default_box_offsets) for a in self.anchors]

    def to_offset_tensor(self, boxes: TLBRBoxes, image_height: int, image_width: int):
        return [a.to_offset_tensor(boxes, image_height, image_width) for a in self.anchors]

    def _build_anchors_at_level(self):
        return [AnchorsAtLevel(level, self.configs) for level in range(self.num_levels)]


class EfficientDetAnchors:

    def __init__(self, size, aspects, num_levels, iou_match_thresh=None):
        self.size = size
        self.num_anchors = len(aspects)
        self.aspects = tf.constant(aspects)
        self.num_levels = num_levels
        self.iou_match_thresh = iou_match_thresh
        tf.debugging.assert_rank(aspects, 2)

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
        out = []
        for level in range(self.num_levels):
            classes, regression = self._build_boxes_for_level(boxes, level)
            regression = EfficientDetAnchors._normalise_regression(regression, boxes)
            out.append(tf.concat([classes, regression], axis=-1))
        return tuple(out)

    def regressions_to_tlbr(self, regressions):
        """

        Parameters
        ----------
        regressions : a list of tensors like [hi, wi, 1 + 4] class + regression

        Returns
        -------

        box tensor : [n, 4] in tlbr format
        label tensor : [n]

        """
        tlbr_boxes = []
        labels = []
        for level, regression in enumerate(regressions):
            regression_label = regression[..., 0]
            regression = regression[..., 1:]
            regression = EfficientDetAnchors._unnormalise_regression(regression, level)
            boxes = self._regress_to_absolute_tlbr(level, regression)
            boxes_out = tf.boolean_mask(boxes, regression_label != efficient_det.NO_CLASS_LABEL)
            labels_out = tf.boolean_mask(regression_label, regression_label != efficient_det.NO_CLASS_LABEL)
            tlbr_boxes.append(boxes_out)
            labels.append(labels_out)
        return tf.concat(tlbr_boxes, axis=0), tf.concat(labels, axis=0)

    def model_out_to_tlbr(self, model_output, thresh=0.5):
        """

        Parameters
        ----------
        regressions : a list of tensors like [hi, wi, 1 + 4] class + regression

        Returns
        -------

        box tensor : [n, 4] in tlbr format
        label tensor : [n]

        """
        regressions = []
        for sub_out in model_output:
            # labels
            label_probs = tf.nn.sigmoid(sub_out[..., :-4])
            best_class = tf.cast(tf.argmax(label_probs, axis=-1)[..., None], tf.float32)
            should_show = tf.reduce_max(label_probs, axis=-1, keepdims=True) > thresh
            reg_label = tf.where(should_show, best_class, efficient_det.NO_CLASS_LABEL)
            # regression
            regression_bboxes = sub_out[..., -4:]
            # join back together
            sub_reg = tf.concat([reg_label, regression_bboxes], axis=-1)
            regressions.append(sub_reg)
        return self.regressions_to_tlbr(regressions)

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

    def _build_boxes_for_level(self, boxes, level):
        return tf.cond(
            tf.size(boxes.box_tensor) == 0,
            lambda: self._empty_level(boxes, level),
            lambda: self._assign_boxes_to_level(boxes, level))

    def _empty_level(self, boxes, level):
        image_height, image_width = boxes.get_image_dimensions()
        default_boxes = self._default_boxes_for_absolute(level, image_height, image_width)
        empty_class = tf.ones(tf.shape(default_boxes)[:-1], dtype=tf.float32)*efficient_det.NO_CLASS_LABEL
        return empty_class[..., None], default_boxes

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
        image_height, image_width = boxes.get_image_dimensions()
        default_boxes = self._default_boxes_for_absolute(level, image_height, image_width)
        best_box_classes, best_ious, best_boxes = boxes.match_with_anchor(default_boxes)
        best_box_classes = tf.where(
            best_ious > self.iou_match_thresh,
            best_box_classes,
            tf.ones_like(best_box_classes)*efficient_det.NO_CLASS_LABEL)
        correspond_regression = EfficientDetAnchors._regression_from_boxes(default_boxes, best_boxes)
        classes = tf.cast(best_box_classes[..., None], tf.float32)
        return classes, correspond_regression

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
        stride = level_to_stride(level)
        pixel_coords = pixel_coordinates(height, width)
        return pixel_coords*stride + stride//2

    def _default_width_height(self, level):
        stride = level_to_stride(level)
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

    def _default_boxes_for_regression(self, level, regression):
        h, w = EfficientDetAnchors._regression_height_width(regression)
        default_boxes = self._default_box_tensor(level, h, w)
        return default_boxes

    def _default_boxes_for_absolute(self, level, image_height, image_width):
        stride = level_to_stride(level)
        regress_height, regress_width = image_height//stride, image_width//stride
        return self._default_box_tensor(level, regress_height, regress_width)

    def num_boxes(self):
        return self.num_anchors

    @staticmethod
    def _normalise_regression(regression, boxes):
        image_height, image_width = boxes.get_image_dimensions()
        norm = EfficientDetAnchors._regression_normalisation(image_height, image_width)
        return regression / norm

    @staticmethod
    def _unnormalise_regression(regression, level):
        h, w = EfficientDetAnchors._regression_height_width(regression)
        stride = level_to_stride(level)
        image_height, image_width = h*stride, w*stride
        norm = EfficientDetAnchors._regression_normalisation(image_height, image_width)
        return regression * norm

    @staticmethod
    def _regression_normalisation(image_height, image_width):
        norm = tf.cast(tf.stack([image_width, image_height], axis=0), tf.float32)
        norm = tf.concat([norm, tf.ones([2])], axis=0)
        return norm[None, None, None]

    @staticmethod
    def _regression_height_width(regression):
        shape = tf.shape(regression)
        if tf.rank(regression) == 5:
            return shape[1], shape[2]
        else:
            return shape[0], shape[1]