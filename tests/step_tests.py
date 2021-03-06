# External Imports
import unittest
import pandas as pd

# Internal Imports
from .utils import rand_df, rand_df_classification
from DSPipeline.errors import TransformError

################################################################################################
# STEP CLASS TESTER
################################################################################################
# This class contains tests that all Step Classes must pass

class StepTest(object):

    ## Make sure that the step class has the correct attributes
    def test_attributes(self):
        self.assertTrue(type(self.step.changes_num_samples) == bool)
        self.assertTrue(type(self.step.description) == str)
        self.assertTrue(len(self.step.description) > 1)

    ## Make sure that the fit function exists
    def test_fit_exists(self):
        self.assertTrue(hasattr(self.step, 'fit') and callable(getattr(self.step, 'fit')))

    ## Make sure that the transform function exists
    def test_transform_exists(self):
        self.assertTrue(hasattr(self.step, 'transform') and callable(getattr(self.step, 'transform')))

    ## Make sure the step works
    def test_step(self):

        # Make sure that transform fails if fit has not been called
        with self.assertRaises(TransformError):
            self.step.transform(self.X, y=self.y)

        # Make sure both fit and transform return a DataFrame
        if self.y is None:
            f_result_X = self.step.fit(self.X)
            t_result_X = self.step.transform(self.X)
            self.assertEqual(type(f_result_X), pd.DataFrame)
            self.assertEqual(type(t_result_X), pd.DataFrame)
        else:
            f_result_X, f_result_y = self.step.fit(self.X, y=self.y)
            t_result_X, t_result_y = self.step.transform(self.X, y=self.y)
            self.assertEqual(type(f_result_X), pd.DataFrame)
            self.assertEqual(type(t_result_X), pd.DataFrame)

        # Test test data transform if applicable
        if not self.step.changes_num_samples:
            new_test = self.step.transform(self.test_X)
            self.assertEqual(type(new_test), pd.DataFrame)

        # If the test appends, check that the returned data frame has more columns than the original
        if hasattr(self.step, 'append_input'):
            if(self.step.append_input):
                self.assertGreater(t_result_X.shape[1], self.X.shape[1])
                if not self.step.changes_num_samples:
                    self.assertGreater(new_test.shape[1], self.test_X.shape[1])
