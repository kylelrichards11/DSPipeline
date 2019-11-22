# External Imports
import unittest
import numpy as np
import pandas as pd

# Internal Imports
from DSPipeline.data_managing import split_x_y
from .utils import rand_df

################################################################################################
# TESTS
################################################################################################
class DataManagingTests(unittest.TestCase):
    
    # Tests that split_x_y works with pandas dataframe
    def test_split_x_y(self):
        X, y = rand_df(shape=(10, 10))
        data = pd.concat((X, y), axis=1)
        X_data, y_data = split_x_y(data, y_label='y')
        self.assertTrue(X.equals(X_data))
        self.assertTrue(y.equals(y_data))

    # Tests that split_x_y raises a type error when not given a dataframe
    def test_split_x_y_raise(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(TypeError):
            split_x_y(data)