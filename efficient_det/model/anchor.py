import tensorflow as tf


class Boxes:
    def __init__(self, image_height, image_width, box_tensor, classes):
        """
        the box tensor should be of the form [nboxes, ymin, xmin, ymax, xmax]

        Parameters
        ----------
        image_height
        image_width
        box_tensor
        classes : [nboxes] tensor no class should have the label -1!!
        """
        self.classes = classes
        self.box_tensor = box_tensor
        self.image_height = image_height
        self.image_width = image_width

    def get_image_dimensions(self):
        return self.image_width, self.image_height

    def boxes_as_centroid_and_widths(self):
        centroid_x = (self.box_tensor[:, 1] + self.box_tensor[:, 3])/2
        centroid_y = (self.box_tensor[:, 0] + self.box_tensor[:, 2])/2
        sx = self.box_tensor[:, 3] - self.box_tensor[:, 1]
        sy = self.box_tensor[:, 2] - self.box_tensor[:, 0]
        return tf.stack([centroid_x, centroid_y, sx, sy], axis=-1)

    @staticmethod
    def _anchor_boxes_as_tlbr_box(anchor_boxes):
        cx, cy, sx, sy = tf.split(anchor_boxes, num_or_size_splits=4, axis=-1)
        ymin = cy - sy
        ymax = cy + sy
        xmin = cx - sx
        xmax = cx + sx
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def match_with_anchor(self, anchor_boxes):
        """

        Parameters
        ----------
        anchor_boxes : tensor of shape [h, w, nanchors, 4]
            represents [cx, cy, sx, sy]

        Returns
        -------
        ious : tensor of shape [..., nboxes]
            the iou for each box

        """
        anchor_boxes_original = Boxes._anchor_boxes_as_tlbr_box(anchor_boxes)
        anchor_original_shape = tf.shape(anchor_boxes_original)
        anchor_h, anchor_w, n_anchors = anchor_original_shape[0], anchor_original_shape[1], anchor_original_shape[2]
        anchor_boxes = tf.reshape(anchor_boxes_original, [-1, 4])

        box_ymin, box_xmin, box_ymax, box_xmax = Boxes.get_box_components(self.box_tensor)
        anc_ymin, anc_xmin, anc_ymax, anc_xmax = Boxes.get_box_components(anchor_boxes)

        # determine the (x, y)-coordinates of the intersection rectangle
        xmin = tf.maximum(box_xmin[:, None], anc_xmin[None, :])
        ymin = tf.maximum(box_ymin[:, None], anc_ymin[None, :])
        xmax = tf.minimum(box_xmax[:, None], anc_xmax[None, :])
        ymax = tf.minimum(box_ymax[:, None], anc_ymax[None, :])
        intersecting_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

        inter_area = Boxes.box_area(intersecting_boxes)
        box_area = Boxes.box_area(self.box_tensor)
        anchor_area = Boxes.box_area(anchor_boxes)
        combo_area = box_area[:, None] + anchor_area[None, :]

        # [nboxes, h*w*nanchors]
        iou = inter_area / (combo_area - inter_area)
        iou = tf.reshape(iou, [-1, anchor_h, anchor_w, n_anchors])
        best_box = tf.argmax(iou, axis=0)

        best_box_class = tf.gather(self.classes, best_box)
        best_iou = tf.gather(iou, best_box)
        best_boxes = tf.gather(self.boxes_as_centroid_and_widths(), best_box)

        return best_box_class, best_iou, best_boxes

    @staticmethod
    def get_box_components(box):
        return tf.split(box, num_or_size_splits=4, axis=-1)

    @staticmethod
    def box_area(boxes):
        ymin, xmin, ymax, xmax = Boxes.get_box_components(boxes)
        return tf.maximum((xmax - xmin + 1), 0) * tf.maximum((ymax - ymin + 1), 0)


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
        boxes : Boxes
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

    def _regress_to_absolute_individual_level(self, level, regression):
        default_boxes = self._default_boxes_for_regression(level, regression)
        offset, scale = tf.split(regression, num_or_size_splits=2, axis=-1)
        centroid, dimensions = tf.split(default_boxes, num_or_size_splits=2, axis=-1)
        regressed_centroids = centroid + offset
        regressed_dimensions = EfficientDetAnchors._scale_width_and_height(dimensions, scale)
        regressions = tf.concat([regressed_centroids, regressed_dimensions], axis=-1)
        return regressions

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
        regressions : tensors of shape [h_i, w_i, n, 4] corresponding to the regressions
        classes : integer tensor of shape [h_i, w_i, n] indication the classes with nan indicating a negative
        """
        default_boxes = self._default_boxes_for_absolute(level, boxes)
        best_box_classes, best_ious, best_boxes = boxes.match_with_anchor(default_boxes)
        best_box_classes = tf.where(best_ious > self.iou_match_thresh, best_box_classes, tf.ones_like(best_box_classes)*-1)
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
        pixel_coords = EfficientDetAnchors._get_pixel_coords(height, width)
        return pixel_coords*EfficientDetAnchors._level_to_stride(level)

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
        regressed_shape = tf.shape(regression)
        default_boxes = self._default_box_tensor(level, regressed_shape[1], regressed_shape[2])
        default_boxes = tf.expand_dims(default_boxes, axis=0)
        return default_boxes

    def _default_boxes_for_absolute(self, level, boxes):
        image_width, image_height = boxes.get_image_dimensions()
        stride = EfficientDetAnchors._level_to_stride(level)
        # todo this may cause some issues
        regress_height, regress_width = image_height//stride, image_width//stride
        return self._default_box_tensor(level, regress_height, regress_width)

    def num_boxes(self):
        return tf.shape(self.aspects)[0]
