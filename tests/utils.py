# External Imports
import numpy as np
import pandas as pd

################################################################################################
# Returns a random dataframe of the given size
def rand_df(shape=(100, 100), val_range=(-100, 100), labeled=True, y_label='label', outlier=False):
    if outlier:
        data = np.random.uniform(low=val_range[0], high=val_range[1], size=(shape[0] - 1, shape[1]))
        outlier = np.random.uniform(low=1e6, high=1e7, size=(1, shape[1]))
        data = np.vstack((data, outlier))
    else:
        data = np.random.uniform(low=val_range[0], high=val_range[1], size=shape)
    cols = [str(i) for i in range(shape[1])]
    if labeled:
        cols[-1] = y_label
    return pd.DataFrame(data, columns=cols)

################################################################################################
# Returns a random dataframe of the given size with class labels
def rand_df_classification(shape=(100, 100), val_range=(-100, 100), y_label='label', classes=2, outlier=False):
    if outlier:
        X_data = np.random.uniform(low=val_range[0], high=val_range[1], size=(shape[0] - 1, shape[1] - 1))
        outlier = np.random.uniform(low=1e6, high=1e7, size=(1, shape[1] - 1))
        X_data = np.vstack((X_data, outlier))
    else:
        X_data = np.random.uniform(low=val_range[0], high=val_range[1], size=(shape[0], shape[1] - 1))
    y_data = np.random.randint(classes, size=(shape[0], 1))
    data = np.hstack((X_data, y_data))
    cols = [str(i) for i in range(shape[1])]
    cols[-1] = y_label
    return pd.DataFrame(data, columns=cols)