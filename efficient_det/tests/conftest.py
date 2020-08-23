import pytest
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 1

import tensorflow as tf


@pytest.fixture
def valid_model_names():
    return list(['b{}'.format(i) for i in range(8)])
