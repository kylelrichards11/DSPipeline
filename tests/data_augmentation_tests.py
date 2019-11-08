# External Imports
import unittest
import numpy as np

# Internal Imports
from DSPipeline.data_augmentation import *
from tests.step_tests import StepTest
from tests.utils import rand_df_classification

TRAIN_SHAPE = (100, 100)
TEST_SHAPE = (100, 99)

################################################################################################
# TESTS
################################################################################################
class ADASYNTests1(unittest.TestCase, StepTest):
    step = ADASYNStep(kwargs={'ratio':{0.0 : 100, 1.0: 100}})
    train_data = rand_df_classification(shape=TRAIN_SHAPE, val_range=(0, 100))
    test_data = rand_df_classification(shape=TEST_SHAPE, val_range=(0, 100))

class SMOTETests1(unittest.TestCase, StepTest):
    step = SMOTEStep()
    train_data = rand_df_classification(shape=TRAIN_SHAPE, val_range=(0, 100))
    test_data = rand_df_classification(shape=TEST_SHAPE, val_range=(0, 100))

