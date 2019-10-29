# External Imports
import unittest

# Internal Imports
from DSPipeline.data_transformations import PCAStep, PolyStep, SinStep, StandardScalerStep
from tests.step_tests import StepTest

################################################################################################
# TESTS
################################################################################################
class PCATests(unittest.TestCase, StepTest):
    step = PCAStep()
    regression = True

class PolyTests(unittest.TestCase, StepTest):
    step = PolyStep()
    regression = True

class SinTests(unittest.TestCase, StepTest):
    step = SinStep()
    regression = True

class StandardScalerTests(unittest.TestCase, StepTest):
    step = StandardScalerStep()
    regression = True