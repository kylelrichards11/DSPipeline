# External Imports
import pandas as pd
import numpy as np

# Internal Imports

################################################################################################
# DATA MANAGING FUNCTIONS
################################################################################################

################################################################################################
# This function combines X and y data into one np array or DataFrame. y_data is always the
# first column
def combine_data(X_data, y_data):
    if type(X_data) == np.ndarray:
        return np.hstack((y_data, X_data))
    elif type(X_data) == pd.DataFrame:
        return pd.concat((y_data, X_data), axis=1)

################################################################################################
# Splits data into X and y components
def split_X_y(data, y_column=0, y_label='label'):
    if type(data) == np.ndarray:
        y_data = np.array(data[:, y_column])
        return np.delete(data, y_column, axis=1), y_data
    elif type(data) == pd.DataFrame:
        y_data = data[y_label]
        return data.drop(y_label, axis=1), y_data
    else:
        raise TypeError(__name__ + ".split_X_y takes numpy array or pandas DataFrame, was " + str(type(data)))
