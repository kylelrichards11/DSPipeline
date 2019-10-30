# External Imports
import unittest
import numpy as np

# Internal Imports
from DSPipeline.data_managing import split_x_y
from .utils import rand_df

################################################################################################
# TESTS
################################################################################################
class DataManagingTests(unittest.TestCase):
    
    # Tests that split_x_y works with pandas dataframe
    def test_split_x_y(self):
        data = rand_df(shape=(10, 10), y_label='y')
        X_data, y_data = split_x_y(data, y_label='y')
        self.assertEqual(X_data.shape, (10, 9))
        self.assertEqual(y_data.shape, (10,))
        self.assertEqual(y_data.name, 'y')

    # Tests that split_x_y raises a type error when not given a dataframe
    def test_split_x_y_raise(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(TypeError):
            split_x_y(data)