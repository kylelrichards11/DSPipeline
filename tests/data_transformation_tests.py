# External Imports
import unittest
import numpy as np

# Internal Imports
from DSPipeline.data_transformations import *
from tests.step_tests import StepTest
from tests.utils import rand_df, rand_df_classification

################################################################################################
# TESTS
################################################################################################
class LDATransform1(unittest.TestCase, StepTest):
    step = LDATransformStep()
    X, y = rand_df_classification()
    test_X = rand_df(labeled=False)

class LDATransform2(unittest.TestCase, StepTest):
    step = LDATransformStep(append_input=True)
    X, y = rand_df_classification()
    test_X = rand_df(labeled=False)

class LogTests1(unittest.TestCase, StepTest):
    step = LogStep()
    X, y = rand_df(val_range=(0, 100))
    test_X = rand_df(val_range=(0, 100), labeled=False)

class LogTests2(unittest.TestCase, StepTest):
    step = LogStep()
    X = rand_df(val_range=(0, 100), labeled=False)
    y = None
    test_X = rand_df(val_range=(0, 100), labeled=False)

class LogTests3(unittest.TestCase, StepTest):
    step = LogStep(append_input=True, columns=['1', '10'], log_func=np.log10)
    X, y = rand_df(val_range=(0, 100))
    test_X = rand_df(val_range=(0, 100), labeled=False)

class PCATests1(unittest.TestCase, StepTest):
    step = PCAStep()
    X, y = rand_df()
    test_X = rand_df(labeled=False)

class PCATests2(unittest.TestCase, StepTest):
    step = PCAStep(append_input=True)
    X = rand_df(labeled=False)
    y = None
    test_X = rand_df(labeled=False)

class PolyTests1(unittest.TestCase, StepTest):
    step = PolyStep()
    X, y = rand_df()
    test_X = rand_df(labeled=False)

class PolyTests2(unittest.TestCase, StepTest):
    step = PolyStep(append_input=True)
    X, y = rand_df(shape=(100, 10))
    test_X = rand_df(shape=(50, 10), labeled=False)

class PolyTests3(unittest.TestCase, StepTest):
    step = PolyStep(append_input=True)
    X = rand_df(shape=(100, 10), labeled=False)
    y = None
    test_X = rand_df(shape=(50, 10), labeled=False)

class SinTests1(unittest.TestCase, StepTest):
    step = SinStep()
    X, y = rand_df()
    test_X = rand_df(labeled=False)

class SinTests2(unittest.TestCase, StepTest):
    step = SinStep()
    X = rand_df(labeled=False)
    y = None
    test_X = rand_df(labeled=False)

class SinTests3(unittest.TestCase, StepTest):
    step = SinStep(append_input=True, columns=['1', '2', '3'])
    X, y = rand_df()
    test_X = rand_df(labeled=False)

class StandardScalerTests1(unittest.TestCase, StepTest):
    step = StandardScalerStep()
    X, y = rand_df()
    test_X = rand_df(labeled=False)

class StandardScalerTests2(unittest.TestCase, StepTest):
    step = StandardScalerStep()
    X = rand_df(labeled=False)
    y = None
    test_X = rand_df(labeled=False)

class StandardScalerTests3(unittest.TestCase, StepTest):
    step = StandardScalerStep(append_input=True)
    X, y = rand_df()
    test_X = rand_df(labeled=False)