# External Imports
import unittest

# Internal Imports
from DSPipeline.outlier_detection import ABODStep, IsoForestStep, LOFStep
from tests.step_tests import StepTest
from tests.utils import rand_df, rand_df_classification

TRAIN_SHAPE = (100, 100)
TEST_SHAPE = (100, 99)

################################################################################################
# TESTS
################################################################################################
class ABODTests(unittest.TestCase, StepTest):
    step = ABODStep(num_remove=1)
    train_data = rand_df(shape=TRAIN_SHAPE, outlier=True)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False, outlier=True)

class IsoForestDefault(unittest.TestCase, StepTest):
    step = IsoForestStep()
    train_data = rand_df(shape=TRAIN_SHAPE, outlier=True)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False, outlier=True)

class IsoForestIncludeY(unittest.TestCase, StepTest):
    step = IsoForestStep(include_y=False)
    train_data = rand_df(shape=TRAIN_SHAPE, outlier=True)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False, outlier=True)

class LFODefault(unittest.TestCase, StepTest):
    step = LOFStep()
    train_data = rand_df_classification(shape=TRAIN_SHAPE, outlier=True)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False, outlier=True)

class LFOIncludeY(unittest.TestCase, StepTest):
    step = LOFStep(include_y=False)
    train_data = rand_df(shape=TRAIN_SHAPE, outlier=True)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False, outlier=True)
