# External Imports
import unittest

# Internal Imports
from DSPipeline.ds_pipeline import EmptyStep, Pipeline
from DSPipeline.feature_selection import PearsonCorrStep
from DSPipeline.outlier_detection import ABODStep
from DSPipeline.errors import TransformError
from tests.step_tests import StepTest
from tests.utils import rand_df

class TestEmptyStep(unittest.TestCase, StepTest):
    step = EmptyStep()
    train_data = rand_df()
    test_data = rand_df(shape=(100, 99))

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.train_data = rand_df(shape=(100, 10))
        self.test_data = rand_df(shape=(100, 9), labeled=False)
        self.steps = [EmptyStep(), PearsonCorrStep(0.1), ABODStep(1)]

    def test_fit(self):
        pipeline = Pipeline(self.steps)
        pipeline.fit(self.train_data)

    def test_transform(self):
        pipeline = Pipeline(self.steps)
        with self.assertRaises(TransformError):
            pipeline.transform(self.test_data)
        try:
            pipeline.transform(self.test_data)
        except TransformError as e:
            self.assertTrue(len(e.message) > 10)
        pipeline.fit(self.train_data)
        pipeline.transform(self.train_data)
        pipeline.transform(self.test_data, allow_sample_removal=False)

    def test_fit_transform(self):
        pipeline = Pipeline(self.steps)
        pipeline.fit_transform(self.train_data, verbose=True)