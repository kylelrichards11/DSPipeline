# External Imports
import unittest
import pandas as pd

# Internal Imports
from .utils import rand_df

################################################################################################
# STEP CLASS TESTER
################################################################################################
# This class contains tests that all Step Classes must pass

class StepTest(object):

    ## Make sure that the step class has the correct attributes
    def test_attributes(self):
        self.assertTrue(type(self.step.removes_samples) == bool)
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

        # Test train data fit transform
        train_data = rand_df()
        self.step.fit(train_data)
        new_train = self.step.transform(train_data)
        self.assertTrue(type(new_train) == pd.DataFrame)

        # Test test data transform if applicable
        if not self.step.removes_samples:
            test_data = rand_df(shape=(100, 99), labeled=False)
            new_test = self.step.transform(test_data)
            self.assertTrue(type(new_train) == pd.DataFrame)