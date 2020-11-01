import tensorflow as tf

from efficient_det.geometry.common import pixel_coordinates


class Boxes:
    def __init__(self, tensor, original_shape=None):
        if original_shape is None:
            original_shape = tensor.shape
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
        intersect_area = tf.reshape(intersect_area, intersecting_boxes.original_shape.as_list()[:1] + [-1])
        combo_area = self.box_area()[:, None] + other.box_area()[None, :]
        iou = intersect_area/(combo_area - intersect_area)
        if as_original_shape:
            iou = tf.reshape(iou, self.original_shape.as_list()[:-1] + [-1])
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
        self.tensor = self.tensor/norm

    def unnormalise(self, height, width):
        norm = TLBRBoxes._build_normalisation_tensor(height, width)
        self.tensor = self.tensor*norm

    def intersecting_boxes(self, other):
        box_a_ymin, box_a_xmin, box_a_ymax, box_a_xmax = self.components()
        box_b_ymin, box_b_xmin, box_b_ymax, box_b_xmax = other.components(transpose=True)

        # determine the (x, y)-coordinates of the intersection rectangle
        xmin = tf.maximum(box_a_xmin, box_b_xmin)
        ymin = tf.maximum(box_a_ymin, box_b_ymin)
        xmax = tf.minimum(box_a_xmax, box_b_xmax)
        ymax = tf.minimum(box_a_ymax, box_b_ymax)
        return TLBRBoxes(tf.stack([ymin, xmin, ymax, xmax], axis=-1))

    def add_offset(self, offset):
        offset = tf.concat([offset, offset], axis=0)[None]
        self.tensor += offset

    def rescale(self, scale):
        self.tensor *= scale

    def boolean_mask(self, mask):
        # this will lose the original shape if it was not a flat list!
        return TLBRBoxes(tf.boolean_mask(self.tensor, mask))

    def box_area(self,):
        # keeps the final 1 dimensions
        ymin, xmin, ymax, xmax = self.components()
        area = tf.maximum((xmax - xmin), 0) * tf.maximum((ymax - ymin), 0)
        return area[..., 0]

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
        centroid_other, dimensions_other = tf.split(other.tensor, num_or_size_splits=2, axis=-1)
        centroid_self, dimensions_self = tf.split(self.tensor, num_or_size_splits=2, axis=-1)
        offset = (centroid_self - centroid_other)/dimensions_other
        scale = tf.math.log(dimensions_self/dimensions_other)
        offset_boxes = tf.concat([offset, scale], axis=-1)
        if as_original_shape:
            offset_boxes = tf.reshape(offset_boxes, self.original_shape)
        return offset_boxes


class DefaultAnchorBoxes(CentroidWidthBoxes):
    def __init__(self, stride_height, stride_width, grid_height, grid_width, box_height, box_width):
        box_shape = tf.stack([box_height/2, box_width/2])
        box_shape = tf.tile(box_shape[None, None], (grid_height, grid_width, 1))
        stride = tf.cast(tf.stack([stride_height, stride_width]), tf.float32)
        stride = tf.tile(stride[None, None], (grid_height, grid_width, 1))
        centroids = pixel_coordinates(grid_height, grid_width)*stride + stride//2
        tensor = tf.concat([centroids, box_shape], axis=-1)
        super(DefaultAnchorBoxes, self).__init__(tensor)


class DefaultAnchorOffsets(Boxes):
    def __init__(self, default_offsets):
        super(DefaultAnchorOffsets, self).__init__(default_offsets)

    def as_centroid_width_box(self, anchors: DefaultAnchorBoxes):
        centroid_offset, dimension_scale = tf.split(self.tensor, num_or_size_splits=2, axis=-1)
        centroid, dimension = tf.split(anchors.tensor, num_or_size_splits=2, axis=-1)
        scale = tf.exp(dimension_scale)
        scaled_dimensions = dimension*scale
        offset_centroids = centroid + centroid_offset*dimension
        offset_anchors = tf.concat([offset_centroids, scaled_dimensions], axis=-1)
        return CentroidWidthBoxes(offset_anchors, self.original_shape)