# External Imports
import unittest

# Internal Imports
from DSPipeline.data_transformations import PCAStep, PolyStep, SinStep, StandardScalerStep
from tests.step_tests import StepTest
from tests.utils import rand_df

TRAIN_SHAPE = (100, 100)
TEST_SHAPE = (100, 99)

################################################################################################
# TESTS
################################################################################################
class PCATests(unittest.TestCase, StepTest):
    step = PCAStep()
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class PolyTests(unittest.TestCase, StepTest):
    step = PolyStep()
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class SinTests(unittest.TestCase, StepTest):
    step = SinStep()
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class StandardScalerTests(unittest.TestCase, StepTest):
    step = StandardScalerStep()
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)