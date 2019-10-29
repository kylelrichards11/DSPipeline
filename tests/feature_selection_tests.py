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

class LassoTests(unittest.TestCase, StepTest):
    step = LassoSelectionStep()

class ListTests(unittest.TestCase, StepTest):
    step = ListSelectionStep(features=['11', '22'])

class PearsonCorrTests(unittest.TestCase, StepTest):
    step = PearsonCorrStep(threshold=0.2)

class RegTreeTests(unittest.TestCase, StepTest):
    step = RegTreeSelectionStep()