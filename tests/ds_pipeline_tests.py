# External Imports
import unittest
import pandas as pd

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
    X, y = rand_df()
    test_X = rand_df(labeled=False)

class TestEmptyStep2(unittest.TestCase, StepTest):
    step = EmptyStep()
    X = rand_df(labeled=False)
    y = None
    test_X = rand_df(labeled=False)

class TestPipeline1(unittest.TestCase):

    def setUp(self):
        self.X, self.y = rand_df(shape=(100, 10))
        self.test_X = rand_df(shape=(100, 10), labeled=False)
        self.steps = [EmptyStep(), PearsonCorrStep(4), PCAStep(kwargs={'n_components':2})]

    def test_fit(self):
        pipeline = Pipeline(self.steps)
        r, _ = pipeline.fit(self.X, y=self.y)
        self.assertEqual(type(r), pd.DataFrame)

    def test_transform(self):
        pipeline = Pipeline(self.steps)
        with self.assertRaises(TransformError):
            pipeline.transform(self.test_X)
        try:
            pipeline.transform(self.test_X)
        except TransformError as e:
            self.assertTrue(len(e.message) > 10)
        r, _ = pipeline.fit(self.X, self.y)
        self.assertEqual(type(r), pd.DataFrame)
        r = pipeline.transform(self.X)
        self.assertEqual(type(r), pd.DataFrame)
        r = pipeline.transform(self.test_X, allow_sample_removal=False)
        self.assertEqual(type(r), pd.DataFrame)

    def test_fit_transform(self):
        pipeline = Pipeline(self.steps)
        r, _ = pipeline.fit_transform(self.X, y=self.y, verbose=True)
        self.assertEqual(type(r), pd.DataFrame)

    def test_pipeline_step(self):
        tr_data_X, tr_data_y = rand_df_classification(shape=(100, 20), classes=3)
        te_data = rand_df(shape=(100, 20), labeled=False)
        scale_step = StandardScalerStep()
        chi_step = ChiSqSelectionStep(select_kwargs={'k':20})
        corr_step = PearsonCorrStep(num_features=0.1)
        pca_step = PCAStep(append_input=False, kwargs={'n_components' : 5})
        poly_step = PolyStep(kwargs={'degree':3, 'include_bias':False})
        pipeline = Pipeline([scale_step, Pipeline([pca_step, poly_step], append_input=True), chi_step])
        r, _ = pipeline.fit_transform(tr_data_X, y=tr_data_y)
        self.assertEqual(type(r), pd.DataFrame)
        r = pipeline.transform(te_data)
        self.assertEqual(type(r), pd.DataFrame)

class TestPipeline2(unittest.TestCase):

    def setUp(self):
        self.X = rand_df(shape=(100, 10), labeled=False)
        self.test_X = rand_df(shape=(100, 10), labeled=False)
        self.steps = [EmptyStep(), StandardScalerStep(append_input=True), PCAStep(kwargs={'n_components':2})]

    def test_fit(self):
        pipeline = Pipeline(self.steps)
        r = pipeline.fit(self.X)
        self.assertEqual(type(r), pd.DataFrame)

    def test_transform(self):
        pipeline = Pipeline(self.steps)
        with self.assertRaises(TransformError):
            pipeline.transform(self.test_X)
        try:
            pipeline.transform(self.test_X)
        except TransformError as e:
            self.assertTrue(len(e.message) > 10)
        r = pipeline.fit(self.X)
        self.assertEqual(type(r), pd.DataFrame)
        r = pipeline.transform(self.X)
        self.assertEqual(type(r), pd.DataFrame)
        r = pipeline.transform(self.test_X, allow_sample_removal=False)
        self.assertEqual(type(r), pd.DataFrame)

    def test_fit_transform(self):
        pipeline = Pipeline(self.steps)
        r = pipeline.fit_transform(self.X, verbose=True)
        self.assertEqual(type(r), pd.DataFrame)

    def test_pipeline_step(self):
        tr_data_X = rand_df(shape=(100, 20), labeled=False)
        te_data = rand_df(shape=(100, 20), labeled=False)
        scale_step = StandardScalerStep()
        pca_step = PCAStep(append_input=False, kwargs={'n_components' : 5})
        poly_step = PolyStep(kwargs={'degree':3, 'include_bias':False})
        pipeline = Pipeline([scale_step, Pipeline([pca_step, poly_step], append_input=True), EmptyStep()])
        r = pipeline.fit_transform(tr_data_X)
        self.assertEqual(type(r), pd.DataFrame)
        r = pipeline.transform(te_data)
        self.assertEqual(type(r), pd.DataFrame)

class TestPipelineStep1(unittest.TestCase, StepTest):
    step = Pipeline([PCAStep(), PolyStep()])
    X, y = rand_df(shape=(100, 10), val_range=(0, 100))
    test_X = rand_df(shape=(100, 10), val_range=(0, 100), labeled=False)

class TestPipelineStep2(unittest.TestCase, StepTest):
    step = Pipeline([PCAStep(), PolyStep()], append_input=True)
    X, y = rand_df(shape=(100, 10), val_range=(0, 100))
    test_X = rand_df(shape=(100, 10), val_range=(0, 100), labeled=False)