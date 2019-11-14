# External Imports
import unittest

# Internal Imports
from DSPipeline.data_transformations import PCAStep, PolyStep, StandardScalerStep
from DSPipeline.ds_pipeline import EmptyStep, Pipeline
from DSPipeline.feature_selection import PearsonCorrStep, ChiSqSelectionStep
from DSPipeline.outlier_detection import ABODStep
from DSPipeline.errors import TransformError
from tests.step_tests import StepTest
from tests.utils import rand_df, rand_df_classification

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

    def test_pipeline_step(self):
        tr_data = rand_df_classification(shape=(100, 21), classes=3)
        te_data = rand_df(shape=(100, 20), labeled=False)
        scale_step = StandardScalerStep()
        chi_step = ChiSqSelectionStep(select_kwargs={'k':20})
        corr_step = PearsonCorrStep(num_features=0.1)
        pca_step = PCAStep(append_input=False, kwargs={'n_components' : 5})
        poly_step = PolyStep(kwargs={'degree':3, 'include_bias':False})
        pipeline = Pipeline([scale_step, Pipeline([pca_step, poly_step], append_input=True), chi_step])
        pipeline.fit_transform(tr_data)
        pipeline.transform(te_data)

class TestPipelineStep1(unittest.TestCase, StepTest):
    step = Pipeline([PCAStep(), PolyStep()])
    train_data = rand_df(shape=(100, 11), val_range=(0, 100))
    test_data = rand_df(shape=(100, 10), val_range=(0, 100), labeled=False)

class TestPipelineStep2(unittest.TestCase, StepTest):
    print("TESTING APPEND")
    step = Pipeline([PCAStep(), PolyStep()], append_input=True)
    train_data = rand_df(shape=(100, 11), val_range=(0, 100))
    test_data = rand_df(shape=(100, 10), val_range=(0, 100), labeled=False)