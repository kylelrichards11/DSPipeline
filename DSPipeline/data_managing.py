# External Imports
import pandas as pd

################################################################################################
def split_x_y(data, y_label='label'):
    """ Splits data into X and y components 
    
    Parameters
    ----------
    data (DataFrame) : the data to split

    y_label (str) : the name of the column containing the y values

    Returns
    -------
    (DataFrame, DataFrame) : a tuple of two DataFrames, the first being the X data and the second being the y data
    
    """
    if type(data) == pd.DataFrame:
        y_data = data[y_label]
        return data.drop(y_label, axis=1), y_data
    else:
        raise TypeError(f'{__name__}.split_x_y pandas DataFrame, was {str(type(data))}')
