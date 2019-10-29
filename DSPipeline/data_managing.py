# External Imports
import pandas as pd

################################################################################################
# Splits data into X and y components
def split_x_y(data, y_label='label'):
    if type(data) == pd.DataFrame:
        y_data = data[y_label]
        return data.drop(y_label, axis=1), y_data
    else:
        raise TypeError(f'{__name__}.split_x_y pandas DataFrame, was {str(type(data))}')
