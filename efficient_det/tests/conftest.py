import pytest
import os
import numpy as np


from pathlib import Path
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 1

import tensorflow as tf


@pytest.fixture
def actual_image():
    testing_dir = Path(__file__).resolve().parent
    test_path = testing_dir.joinpath('testImages/test.jpg')
    return tf.image.convert_image_dtype(np.asarray(Image.open(test_path)), tf.float32)


@pytest.fixture
def valid_model_names():
    return list(['b{}'.format(i) for i in range(8)])


