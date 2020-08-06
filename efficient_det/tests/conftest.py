import pytest
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf


@pytest.fixture
def valid_model_names():
    return list(['b{}'.format(i) for i in range(8)])
