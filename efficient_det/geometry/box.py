import tensorflow as tf

from efficient_det.geometry.common import pixel_coordinates


class Boxes:
    def __init__(self, tensor, original_shape=None):
        if original_shape is None:
            original_shape = tf.shape(tensor)
        self.original_shape = original_shape
        self.tensor = tf.reshape(tensor, [-1, 4])

    def is_empty(self):
        return tf.shape(self.tensor)[0] == 0

    def as_original_shape(self,):
        return tf.reshape(self.tensor, self.original_shape)

    def components(self, transpose=False):
        components = []
        tensor = tf.transpose(self.tensor) if transpose else self.tensor
        for i in range(4):
            comp = tensor[i:i+1, ...] if transpose else tensor[..., i:i+1]
            components.append(comp)
        return components

    def get_tensor(self):
        return self.tensor


class TLBRBoxes(Boxes):
    def __init__(self, tensor, original_shape=None):
        super(TLBRBoxes, self).__init__(tensor, original_shape)

    def iou(self, other, as_original_shape=False):
        intersecting_boxes = self.intersecting_boxes(other)
        intersect_area = intersecting_boxes.box_area()
        combo_area = self.box_area()[:, None] + other.box_area()[None, :]
        iou = intersect_area/(combo_area - intersect_area)
        if as_original_shape:
            iou = tf.reshape(iou, self.original_shape.as_list() + [-1])
        return iou

    def as_centroid_and_width_box(self):
        centroid_x = (self.tensor[..., 1] + self.tensor[..., 3]) / 2
        centroid_y = (self.tensor[..., 0] + self.tensor[..., 2]) / 2
        sx = (self.tensor[..., 3] - self.tensor[..., 1]) / 2
        sy = (self.tensor[..., 2] - self.tensor[..., 0]) / 2
        box_coords = tf.stack([centroid_x, centroid_y, sx, sy], axis=-1)
        return CentroidWidthBoxes(box_coords, self.original_shape)

    def normalise(self, height, width):
        norm = TLBRBoxes._build_normalisation_tensor(height, width)
        self.tlbr = self.tlbr/norm

    def unnormalise(self, height, width):
        norm = TLBRBoxes._build_normalisation_tensor(height, width)
        self.tlbr = self.tlbr*norm

    def intersecting_boxes(self, other):
        box_a_ymin, box_a_xmin, box_a_ymax, box_a_xmax = self.components()
        box_b_ymin, box_b_xmin, box_b_ymax, box_b_xmax = other.components(transpose=True)

        # determine the (x, y)-coordinates of the intersection rectangle
        xmin = tf.maximum(box_a_ymin, box_b_xmin)
        ymin = tf.maximum(box_a_ymin, box_b_ymin)
        xmax = tf.minimum(box_a_xmax, box_b_xmax)
        ymax = tf.minimum(box_a_ymax, box_b_ymax)

        return TLBRBoxes(tf.stack([ymin, xmin, ymax, xmax], axis=-1), self.original_shape)

    def box_area(self):
        ymin, xmin, ymax, xmax = self.components()
        return tf.maximum((xmax - xmin), 0) * tf.maximum((ymax - ymin), 0)

    def create_box_tensor_from_indices(self, indices):
        return tf.gather(self.tensor, indices)

    @staticmethod
    def _build_normalisation_tensor(height, width):
        tensor = tf.stack([height, width, height, width])
        return tf.cast(tensor, tf.float32)


class CentroidWidthBoxes(Boxes):
    def __init__(self, tensor, original_shape=None):
        super(CentroidWidthBoxes, self).__init__(tensor, original_shape)

    def as_tlbr_box(self):
        cx, cy, sx, sy = self.components()
        ymin = cy - sy
        ymax = cy + sy
        xmin = cx - sx
        xmax = cx + sx
        return TLBRBoxes(tf.concat([ymin, xmin, ymax, xmax], axis=-1), self.original_shape)

    def as_offset_boxes(self, other, as_original_shape=False):
        centroid_self, dimension_self = tf.split(other.tensor, num_or_size_splits=2, axis=-1)
        centroid_other, dimensions_other = tf.split(self.tensor, num_or_size_splits=2, axis=-1)
        offset = centroid_other - centroid_self
        scale = tf.math.log(dimensions_other / dimension_self)
        offset_boxes = tf.concat([offset, scale], axis=-1)
        if as_original_shape:
            offset_boxes = tf.reshape(offset_boxes, self.original_shape)
        return offset_boxes


class DefaultAnchorBoxes(CentroidWidthBoxes):
    def __init__(self, stride_height, stride_width, grid_height, grid_width, box_height, box_width):
        box_shape = tf.constant([box_height/2, box_width/2])
        box_shape = tf.tile(box_shape[None, None], (grid_height, grid_width, 1))
        stride = tf.constant([stride_height,stride_width])
        stride = tf.tile(stride[None, None], (grid_height, grid_width, 1))
        centroids = pixel_coordinates(grid_height, grid_width)*stride + stride//2
        tensor = tf.concat([centroids, box_shape], axis=-1)
        super(DefaultAnchorBoxes, self).__init__(tensor)


class DefaultAnchorOffsets(Boxes):
    def __init__(self, default_offsets):
        super(DefaultAnchorOffsets, self).__init__(default_offsets)

    def as_centroid_width_box(self, anchors: DefaultAnchorBoxes, as_original_shape=False):
        centroid_offset, dimension_scale = tf.split(self.tensor, num_or_size_splits=2, axis=-1)
        centroid, dimension = tf.split(anchors, num_or_size_splits=2, axis=-1)
        width_height_scale = tf.exp(dimension_scale)

        scaled_dimensions = dimension*width_height_scale
        offset_centroids = centroid + centroid_offset
        offset_anchors = tf.concat([offset_centroids, scaled_dimensions], axis=-1)
        if as_original_shape:
            offset_anchors = tf.reshape(offset_anchors, self.original_shape)
        return CentroidWidthBoxes(offset_anchors)




class Boxes:

    def __init__(self, image_height, image_width, box_tensor, classes):
        """
        the box tensor should be of the form [nboxes, ymin, xmin, ymax, xmax]
        in normal or unormal doesnt matter, except for area!

        Parameters
        ----------
        image_height
        image_width
        box_tensor
        classes : [nboxes] tensor no class should have the label -1!!
        """
        self.classes = classes
        self.box_tensor = tf.cast(box_tensor, tf.float32)
        self.image_height = image_height
        self.image_width = image_width

    def unnormalise(self):
        unnormalisation = self._normalisation_tensor()
        self.box_tensor *= unnormalisation
        return self.box_tensor

    def normalise(self,):
        normalisation = self._normalisation_tensor()
        self.box_tensor /= normalisation
        return self.box_tensor

    def are_empty(self):
        return tf.shape(self.box_tensor)[0] == 0

    def _normalisation_tensor(self):
        tensor = tf.stack([self.image_height, self.image_width, self.image_height, self.image_width])[None]
        return tf.cast(tensor, tf.float32)

    def image_dim_normalisation_tensor(self):
        tensor = tf.stack([self.image_width, self.image_height])
        return tf.cast(tensor, tf.float32)

    def get_image_dimensions(self):
        return self.image_height, self.image_width

    def boxes_as_centroid_and_widths(self):
        centroid_x = (self.box_tensor[:, 1] + self.box_tensor[:, 3])/2
        centroid_y = (self.box_tensor[:, 0] + self.box_tensor[:, 2])/2
        sx = (self.box_tensor[:, 3] - self.box_tensor[:, 1])/2
        sy = (self.box_tensor[:, 2] - self.box_tensor[:, 0])/2
        return tf.stack([centroid_x, centroid_y, sx, sy], axis=-1)

    @staticmethod
    def absolute_as_tlbr(anchor_boxes):
        cx, cy, sx, sy = tf.split(anchor_boxes, num_or_size_splits=4, axis=-1)
        ymin = cy - sy
        ymax = cy + sy
        xmin = cx - sx
        xmax = cx + sx
        return tf.concat([ymin, xmin, ymax, xmax], axis=-1)

    def match_with_anchor(self, anchor_boxes):
        """
        taken from https://github.com/venuktan/Intersection-Over-Union/blob/master/iou_benchmark.py

        Parameters
        ----------
        anchor_boxes : tensor of shape [h, w, nanchors, 4]
            represents [cx, cy, sx, sy]

        Returns
        -------
        ious : tensor of shape [..., nboxes]
            the iou for each box

        """
        anchor_boxes_original = Boxes.absolute_as_tlbr(anchor_boxes)
        anchor_original_shape = tf.shape(anchor_boxes_original)
        anchor_h, anchor_w, n_anchors = anchor_original_shape[0], anchor_original_shape[1], anchor_original_shape[2]
        anchor_boxes = tf.reshape(anchor_boxes_original, [-1, 4])

        # intersect boxes and get area
        intersecting_boxes = Boxes.intersecting_boxes(self.box_tensor, anchor_boxes)
        inter_area = Boxes.box_area(intersecting_boxes)
        box_area = Boxes.box_area(self.box_tensor)
        anchor_area = Boxes.box_area(anchor_boxes)
        combo_area = box_area[:, None] + anchor_area[None, :]

        # [nboxes, h*w*nanchors]
        iou = inter_area / (combo_area - inter_area)
        iou = tf.reshape(iou, [-1, anchor_h, anchor_w, n_anchors])
        best_box = tf.argmax(iou, axis=0)

        best_box_class = tf.gather(self.classes, best_box)
        best_iou = tf.reduce_max(iou, axis=0)
        best_boxes = tf.gather(self.boxes_as_centroid_and_widths(), best_box)

        return best_box_class, best_iou, best_boxes

    @staticmethod
    def get_box_components(box):
        return [x[..., 0] for x in tf.split(box, num_or_size_splits=4, axis=-1, )]

    @staticmethod
    def intersecting_boxes(box_a, box_b):
        box_a_ymin, box_a_xmin, box_a_ymax, box_a_xmax = Boxes.get_box_components(box_a)
        box_b_ymin, box_b_xmin, box_b_ymax, box_b_xmax = Boxes.get_box_components(box_b)

        # determine the (x, y)-coordinates of the intersection rectangle
        xmin = tf.maximum(box_a_xmin[:, None], box_b_xmin[None, :])
        ymin = tf.maximum(box_a_ymin[:, None], box_b_ymin[None, :])
        xmax = tf.minimum(box_a_xmax[:, None], box_b_xmax[None, :])
        ymax = tf.minimum(box_a_ymax[:, None], box_b_ymax[None, :])
        intersecting_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        return intersecting_boxes

    @staticmethod
    def box_area(boxes):
        ymin, xmin, ymax, xmax = Boxes.get_box_components(boxes)
        return tf.maximum((xmax - xmin), 0) * tf.maximum((ymax - ymin), 0)

    @staticmethod
    def from_image_and_boxes(image, bboxes):
        image_shape = tf.shape(image)
        bbox = Boxes(
            image_height=image_shape[0],
            image_width=image_shape[1],
            box_tensor=bboxes,
            classes=None)
        return bbox

    @staticmethod
    def from_image_boxes_labels(image, bboxes, labels):
        image_shape = tf.shape(image)
        bbox = Boxes(
            image_height=image_shape[0],
            image_width=image_shape[1],
            box_tensor=bboxes,
            classes=labels)
        return bbox