# External Imports
import unittest
import numpy as np

# Internal Imports
from DSPipeline.data_augmentation import *
from tests.step_tests import StepTest
from tests.utils import rand_df_classification, rand_df

################################################################################################
# TESTS
################################################################################################
class ADASYNTests1(unittest.TestCase, StepTest):
    step = ADASYNStep(kwargs={'ratio':{0.0 : 100, 1.0: 100}})
    X, y = rand_df_classification(val_range=(0, 100))
    test_X = rand_df(val_range=(0, 100))

class SMOTETests1(unittest.TestCase, StepTest):
    step = SMOTEStep()
    X, y = rand_df_classification(val_range=(0, 100))
    test_X = rand_df(val_range=(0, 100))

