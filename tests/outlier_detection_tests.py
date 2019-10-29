# External Imports
import unittest

# Internal Imports
from DSPipeline.outlier_detection import ABODStep, IsoForestStep, LOFStep
from tests.step_tests import StepTest

################################################################################################
# TESTS
################################################################################################
class ABODTests(unittest.TestCase, StepTest):
    step = ABODStep(num_remove=1)

class IsoForestTests(unittest.TestCase, StepTest):
    step = IsoForestStep()

class LFOTests(unittest.TestCase, StepTest):
    step = LOFStep()
