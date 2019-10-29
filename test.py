################################################################################################
# TESTING
################################################################################################

# External Imports
import unittest

# Internal Imports
from tests.data_transformation_tests import PCATests, PolyTests, SinTests, StandardScalerTests
from tests.feature_selection_tests import ChiSqTests, LassoTests, ListTests, PearsonCorrTests, RegTreeTests
from tests.outlier_detection_tests import ABODTests, IsoForestTests, LFOTests

if __name__ == "__main__":
    unittest.main()