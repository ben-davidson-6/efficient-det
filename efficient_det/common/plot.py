import matplotlib.pyplot
import tensorflow as tf

from efficient_det.common.box import Boxes
from efficient_det.model.anchor import EfficientDetAnchors


class Plotter:
    COLORS = [
        (1., 0., 0.),
        (0., 1., 0.),
        (0., 0., 1.),
    ]

    def __init__(self, image, boxes: Boxes, normalised=False):
        self.boxes = boxes
        if not normalised:
            self.boxes.normalise()
        if image.dtype == tf.uint8:
            image = tf.image.convert_image_dtype(image, tf.float32)
        self.image = image

    def draw_boxes(self):
        # todo this could also add labels
        bboxes = tf.image.draw_bounding_boxes(
            self.image[None],
            self.boxes.box_tensor[None],
            Plotter.COLORS)[0]
        return bboxes

    def plot(self, subplot=None, title='', plt=None):
        if plt is None:
            plt = matplotlib.pyplot

        if subplot:
            ax = plt.subplot(*subplot)
            if title:
                ax.title.set_text(title)
        else:
            if title:
                plt.suptitle(title, wrap=True)
        plt.imshow(self.draw_boxes())
        plt.axis('off')

    @staticmethod
    def from_anchor_and_regresion(image, regressions, anchor: EfficientDetAnchors):
        boxes, labels = anchor.regressions_to_tlbr(regressions)
        boxes = Boxes.from_image_boxes_labels(image, boxes, labels)
        return Plotter(image, boxes)


