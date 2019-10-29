# External Imports
import numpy as np
import pandas as pd

################################################################################################
# Returns a random dataframe of the given size
def rand_df(shape=(100, 100), labeled=True, y_label='label'):
    data = np.random.uniform(low=-100, high=100, size=shape)
    cols = [str(i) for i in range(shape[1])]
    if labeled:
        cols[-1] = y_label
    return pd.DataFrame(data, columns=cols)

################################################################################################
# Returns a random dataframe of the given size with class labels
def rand_df_classification(shape=(100, 100), y_label='label', classes=2):
    X_data = np.random.uniform(low=-100, high=100, size=(shape[0], shape[1]-1))
    y_data = np.random.randint(classes, size=(shape[0], 1))
    data = np.hstack((X_data, y_data))
    cols = [str(i) for i in range(shape[1])]
    cols[-1] = y_label
    return pd.DataFrame(data, columns=cols)