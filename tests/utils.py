# External Imports
import numpy as np
import pandas as pd

################################################################################################
# Returns a random dataframe of the given size
def rand_df(shape=(100, 100), val_range=(-100, 100), labeled=True, outlier=False):
    if outlier:
        X = np.random.uniform(low=val_range[0], high=val_range[1], size=(shape[0] - 1, shape[1]))
        outlier = np.random.uniform(low=1e6, high=1e7, size=(1, shape[1]))
        X = np.vstack((X, outlier))
    else:
        X = np.random.uniform(low=val_range[0], high=val_range[1], size=shape)
    cols = [str(i) for i in range(shape[1])]
    if labeled:
        y = np.random.uniform(low=val_range[0], high=val_range[1], size=(shape[0],))
        return pd.DataFrame(X, columns=cols), pd.Series(y, name='y')
    return pd.DataFrame(X, columns=cols)

################################################################################################
# Returns a random dataframe of the given size with class labels
def rand_df_classification(shape=(100, 100), val_range=(-100, 100), classes=2, outlier=False):
    X = rand_df(shape=shape, val_range=val_range, outlier=outlier, labeled=False)
    y = np.random.randint(classes, size=(shape[0],))
    return X, pd.Series(y, name='y')