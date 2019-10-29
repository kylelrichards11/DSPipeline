# External Imports
import unittest

# Internal Imports
from DSPipeline.feature_selection import ChiSqSelectionStep, LassoSelectionStep, ListSelectionStep, PearsonCorrStep, RegTreeSelectionStep
from tests.step_tests import StepTest

################################################################################################
# TESTS
################################################################################################
class ChiSqTests(unittest.TestCase, StepTest):
    step = ChiSqSelectionStep()
    regression = False

class LassoTests(unittest.TestCase, StepTest):
    step = LassoSelectionStep()
    regression = True

class ListTests(unittest.TestCase, StepTest):
    step = ListSelectionStep(features=['11', '22'])
    regression = True

class PearsonCorrTests(unittest.TestCase, StepTest):
    step = PearsonCorrStep(threshold=0.2)
    regression = True

class RegTreeTests(unittest.TestCase, StepTest):
    step = RegTreeSelectionStep()
    regression = True