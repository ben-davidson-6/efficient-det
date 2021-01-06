from efficient_det.constants import *

from tensorflow.keras.mixed_precision import experimental as mixed_precision

USE_MIXED = True

if USE_MIXED:
    print('using mixed precision')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)