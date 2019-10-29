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