################################################################################################
# TESTING
################################################################################################

# External Imports
import unittest

# Internal Imports
from tests.data_manager_tests import DataManagingTests
from tests.data_transformation_tests import PCATests, PolyTests, SinTests, StandardScalerTests
from tests.ds_pipeline_tests import TestEmptyStep, TestPipeline
from tests.feature_selection_tests import ChiSqTests, LassoTests, ListTests, PearsonCorrTests, RegTreeTests
from tests.outlier_detection_tests import ABODTests, IsoForestDefault, IsoForestIncludeY, LFODefault, LFOIncludeY

if __name__ == "__main__":
    unittest.main()