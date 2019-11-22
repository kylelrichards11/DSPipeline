# External Imports
import unittest

# Internal Imports
from DSPipeline.outlier_detection import ABODStep, IsoForestStep, LOFStep
from tests.step_tests import StepTest
from tests.utils import rand_df, rand_df_classification

################################################################################################
# TESTS
################################################################################################
class ABODTests(unittest.TestCase, StepTest):
    step = ABODStep(num_remove=1)
    X, y = rand_df_classification(outlier=True)
    test_X = rand_df(labeled=False, outlier=True)

class IsoForestDefault(unittest.TestCase, StepTest):
    step = IsoForestStep()
    X, y = rand_df(outlier=True)
    test_X = rand_df(labeled=False, outlier=True)

class IsoForestIncludeY(unittest.TestCase, StepTest):
    step = IsoForestStep(include_y=False)
    X, y = rand_df(outlier=True)
    test_X = rand_df(labeled=False, outlier=True)

class LFODefault(unittest.TestCase, StepTest):
    step = LOFStep()
    X, y = rand_df_classification(outlier=True)
    test_X = rand_df(labeled=False, outlier=True)

class LFOIncludeY(unittest.TestCase, StepTest):
    step = LOFStep(include_y=False)
    X, y = rand_df(outlier=True)
    test_X = rand_df(labeled=False, outlier=True)
