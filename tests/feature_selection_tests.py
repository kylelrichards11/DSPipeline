# External Imports
import unittest

# Internal Imports
from DSPipeline.feature_selection import ChiSqSelectionStep, LassoSelectionStep, ListSelectionStep, PearsonCorrStep, RegTreeSelectionStep
from tests.step_tests import StepTest
from tests.utils import rand_df, rand_df_classification

TRAIN_SHAPE = (100, 100)
TEST_SHAPE = (100, 99)

################################################################################################
# TESTS
################################################################################################
class ChiSqTests(unittest.TestCase, StepTest):
    step = ChiSqSelectionStep()
    train_data = rand_df_classification(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class LassoTests(unittest.TestCase, StepTest):
    step = LassoSelectionStep()
    train_data = rand_df_classification(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class ListTests(unittest.TestCase, StepTest):
    step = ListSelectionStep(features=['11', '22'])
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

    # Tests that a key error is raised when the features do not exist
    def test_key_error(self):
        s = ListSelectionStep(features=['11', 'not_a_feature'])
        data = rand_df(shape=(10, 15))
        with self.assertRaises(KeyError):
            s.fit(data)

class PearsonCorrTests1(unittest.TestCase, StepTest):
    step = PearsonCorrStep(num_features=0.2)
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class PearsonCorrTests2(unittest.TestCase, StepTest):
    step = PearsonCorrStep(num_features=50)
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)

class RegTreeTests(unittest.TestCase, StepTest):
    step = RegTreeSelectionStep()
    train_data = rand_df(shape=TRAIN_SHAPE)
    test_data = rand_df(shape=TEST_SHAPE, labeled=False)