# External Imports
import unittest
import numpy as np

# Internal Imports
from DSPipeline.data_transformations import *
from tests.step_tests import StepTest
from tests.utils import rand_df

TRAIN_SHAPE = (100, 100)
TEST_SHAPE = (100, 99)

################################################################################################
# TESTS
################################################################################################
class LogTests1(unittest.TestCase, StepTest):
    step = LogStep()
    train_data = rand_df(shape=TRAIN_SHAPE, val_range=(0, 100))
    test_data = rand_df(shape=TEST_SHAPE, val_range=(0, 100), labeled=False)

class LogTests2(unittest.TestCase, StepTest):
    step = LogStep(append_input=True, columns=['1', '10'], log_func=np.log10)
    train_data = rand_df(shape=TRAIN_SHAPE, val_range=(0, 100))
    test_data = rand_df(shape=TEST_SHAPE, val_range=(0, 100), labeled=False)

class PCATests1(unittest.TestCase, StepTest):
    step = PCAStep()
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class PCATests2(unittest.TestCase, StepTest):
    step = PCAStep(append_input=True)
    train_data = rand_df(shape=TRAIN_SHAPE, labeled=False)
    test_data = rand_df(shape=TRAIN_SHAPE, labeled=False)

class PolyTests1(unittest.TestCase, StepTest):
    step = PolyStep()
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class PolyTests2(unittest.TestCase, StepTest):
    step = PolyStep(append_input=True)
    train_data = rand_df(shape=(100, 10))
    test_data = rand_df(shape=(50, 9), labeled=False)

class SinTests1(unittest.TestCase, StepTest):
    step = SinStep()
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class SinTests2(unittest.TestCase, StepTest):
    step = SinStep(append_input=True, columns=['1', '2', '3'])
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class StandardScalerTests1(unittest.TestCase, StepTest):
    step = StandardScalerStep()
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class StandardScalerTests2(unittest.TestCase, StepTest):
    step = StandardScalerStep()
    train_data = rand_df(shape=TRAIN_SHAPE, labeled=False)
    test_data = rand_df(shape=TRAIN_SHAPE, labeled=False)

class StandardScalerTests3(unittest.TestCase, StepTest):
    step = StandardScalerStep(append_input=True)
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)