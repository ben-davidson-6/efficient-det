import tensorflow as tf

from efficient_det.geometry.box import DefaultAnchorBoxes, TLBRBoxes, DefaultAnchorOffsets, CentroidWidthBoxes
from efficient_det.geometry.common import level_to_stride


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
        self.anchor_builders = [SingleAnchorAtLevelBuilder(config, level) for config in configs]

    def match_boxes(self, boxes: TLBRBoxes, image_height: int, image_width: int):
        anchors = self._build_anchors(image_height, image_width)
        ious = anchors.iou(boxes, as_original_shape=True)
        max_iou = tf.reduce_max(ious, axis=-1)
        best_box = tf.argmax(ious, axis=-1)
        box_coordinates = boxes.create_box_tensor_from_indices(best_box)
        return TLBRBoxes(box_coordinates), max_iou, best_box

    def to_tlbr_tensor(self, default_box_offsets: tf.Tensor):
        """regression should be height, width, n_anchors, 4"""
        image_height, image_width = self._original_image_size_from_default_offsets(default_box_offsets)
        anchors = self._build_anchors(image_height, image_width)
        offsets = DefaultAnchorOffsets(default_box_offsets)
        centroid_widths = offsets.as_centroid_width_box(anchors)
        return centroid_widths.as_tlbr_box().as_original_shape()

    def to_offset_tensor(self, boxes: TLBRBoxes, image_height: int, image_width: int):
        box_coordinates, ious, best_box = self.match_boxes(boxes, image_height, image_width)
        anchors = self._build_anchors(image_height, image_width)
        centroid_boxes = box_coordinates.as_centroid_and_width_box()
        offset = centroid_boxes.as_offset_boxes(anchors, as_original_shape=True)
        return offset, ious, best_box

    def _build_anchors(self, image_height, image_width):
        anchor_tensors = tf.stack([a.build_anchors(image_height, image_width).as_original_shape() for a in self.anchor_builders], axis=-2)
        return CentroidWidthBoxes(anchor_tensors).as_tlbr_box()

    def _original_image_size_from_default_offsets(self, offset):
        offset_shape = tf.shape(offset)
        image_height = level_to_stride(self.level)*offset_shape[1]
        image_width = level_to_stride(self.level)*offset_shape[2]
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
        normalisation = tf.stack([image_height, image_width])
        normalised_box_shape = box_shape / tf.cast(normalisation, tf.float32)
        return normalised_box_shape[0], normalised_box_shape[1]


class Anchors:
    def __init__(self, num_levels, configs):
        self.num_levels = num_levels
        self.aspects = configs
        self.anchors = self._build_anchors_at_level()

    def to_tlbr_tensor(self, default_box_offsets: tuple):
        assert len(default_box_offsets) == len(self.anchors)
        return [a.to_tlbr_tensor(offset) for a, offset in zip(self.anchors, default_box_offsets)]

    def to_offset_tensors(self, boxes: TLBRBoxes, image_height: int, image_width: int):
        return [a.to_offset_tensor(boxes, image_height, image_width) for a in self.anchors]

    def _build_anchors_at_level(self):
        return [AnchorsAtLevel(level, self.aspects) for level in range(self.num_levels)]


def build_anchors(size, num_levels, aspects):
    configs = []
    for aspect in aspects:
        config = SingleAnchorConfig(size, tf.constant(aspect))
        configs.append(config)
    return Anchors(num_levels, configs)