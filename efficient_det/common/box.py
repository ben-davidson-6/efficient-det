import tensorflow as tf


def check_normalised(f):
    def deco_f(self):
        gt = tf.reduce_all(tf.greater_equal(self.box_tensor, 0))
        lt = tf.reduce_all(tf.less_equal(self.box_tensor, 1))
        if not tf.logical_and(gt, lt):
            tf.print('WARNING you are trying to normalise something which looks normal')
        f(self)
    return deco_f


def check_unnormalised(f):
    def deco_f(self):
        gt = tf.reduce_any(tf.greater(self.box_tensor, 1))
        if not gt:
            tf.print('WARNING you are trying to unnormalise something which looks unnormal')
        f(self)
    return deco_f


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

    @check_normalised
    def unnormalise(self):
        unnormalisation = self._normalisation_tensor()
        self.box_tensor *= unnormalisation
        return self.box_tensor

    @check_unnormalised
    def normalise(self,):
        normalisation = self._normalisation_tensor()
        self.box_tensor /= normalisation
        return self.box_tensor

    def _normalisation_tensor(self):
        tensor = tf.stack([self.image_height, self.image_width, self.image_height, self.image_width])[None]
        return tf.cast(tensor, tf.float32)

    def get_image_dimensions(self):
        return self.image_width, self.image_height

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