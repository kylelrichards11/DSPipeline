# External Imports
import unittest
from sklearn.ensemble import ExtraTreesClassifier

# Internal Imports
from DSPipeline.feature_selection import ChiSqSelectionStep, LassoSelectionStep, ListSelectionStep, PearsonCorrStep, TreeSelectionStep
from tests.step_tests import StepTest
from tests.utils import rand_df, rand_df_classification

################################################################################################
# TESTS
################################################################################################
class ChiSqTests(unittest.TestCase, StepTest):
    step = ChiSqSelectionStep()
    X, y = rand_df_classification()
    test_X = rand_df(labeled=False)

class LassoTests(unittest.TestCase, StepTest):
    step = LassoSelectionStep()
    X, y = rand_df_classification()
    test_X = rand_df(labeled=False)

class ListTests(unittest.TestCase, StepTest):
    step = ListSelectionStep(features=['11', '22'])
    X, y = rand_df()
    test_X = rand_df(labeled=False)

    # Tests that a key error is raised when the features do not exist
    def test_key_error(self):
        s = ListSelectionStep(features=['11', 'not_a_feature'])
        tr_X, tr_y = rand_df(shape=(10, 15))
        with self.assertRaises(KeyError):
            s.fit(tr_X, y=tr_y)

class ListTests2(unittest.TestCase, StepTest):
    step = ListSelectionStep(features=['11', '22'])
    X = rand_df(labeled=False)
    y = None
    test_X = rand_df(labeled=False)

class PearsonCorrTests1(unittest.TestCase, StepTest):
    step = PearsonCorrStep(num_features=0.2)
    X, y = rand_df()
    test_X = rand_df(labeled=False)

class PearsonCorrTests2(unittest.TestCase, StepTest):
    step = PearsonCorrStep(num_features=50)
    X, y = rand_df()
    test_X = rand_df(labeled=False)

class TreeTests1(unittest.TestCase, StepTest):
    step = TreeSelectionStep()
    X, y = rand_df()
    test_X = rand_df(labeled=False)

class TreeTests2(unittest.TestCase, StepTest):
    step = TreeSelectionStep(tree_model=ExtraTreesClassifier)
    X, y = rand_df_classification()
    test_X = rand_df(labeled=False)