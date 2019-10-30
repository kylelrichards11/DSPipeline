# External Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
import numpy as np

# Internal Imports
from .data_managing import split_x_y
from .errors import TransformError

################################################################################################
# SCALE DATA
################################################################################################

class StandardScalerStep():
    def __init__(self, kwargs={}):
        self.description = "Standard Scaler"
        self.kwargs = kwargs
        self.fitted = None
        self.removes_samples = False

    def fit(self, data, y_label='label'):
        if y_label in data.columns:
            X_data, y_data = split_x_y(data, y_label=y_label)
        else:
            X_data = data
            y_data = None
        scaler = StandardScaler(**self.kwargs)  
        self.fitted = scaler.fit(X_data)
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.fitted is None:
            raise TransformError

        if y_label in data.columns:
            X_data, y_data = split_x_y(data, y_label=y_label)
        else:
            X_data = data
            y_data = None

        X_scaled = pd.DataFrame(self.fitted.transform(X_data), columns=X_data.columns)
        if y_data is not None:
            return pd.concat((X_scaled, y_data), axis=1)
        return X_scaled

################################################################################################
# PCA
################################################################################################
# Principal Component Analysis with or without appending principal components to original data

class PCAStep():
    def __init__(self, append_data=False, kwargs={}):
        self.description = 'PCA'
        self.kwargs = kwargs
        self.append_data = append_data
        self.fitted = None
        self.removes_samples = False

    def fit(self, data, y_label='label'):

        if y_label in data.columns:
            X_data, y_data = split_x_y(data, y_label=y_label)
        else:
            X_data = data
            y_data = None

        pca_model = PCA(**self.kwargs)
        self.fitted = pca_model.fit(X_data)
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.fitted is None:
            raise TransformError
        
        if y_label in data.columns:
            X_data, y_data = split_x_y(data, y_label=y_label)
        else:
            X_data = data
            y_data = None

        pca_data = self.fitted.transform(X_data)
        
        # Get column names for post pca dataframe
        cols = []
        for i in range(1, pca_data.shape[1]+1):
            cols.append("PC_" + str(i))

        # Return pca data with or without appending
        if self.append_data:
            return pd.concat((data, pd.DataFrame(pca_data, columns=cols)), axis=1)
        if y_data is None:
            return pd.DataFrame(pca_data, columns=cols)
        return pd.concat((y_data, pd.DataFrame(pca_data, columns=cols)), axis=1)

################################################################################################
# POLYNOMIAL INTERACTIONS FEATURES
################################################################################################
# Generates interaction terms and polynomials with or without appending to original data

class PolyStep():

    def __init__(self, append_data=False, kwargs={}):
        self.description = 'Polynomial Features'
        self.kwargs = kwargs
        self.append_data = append_data
        self.fitted = None
        self.removes_samples = False

    def fit(self, data, y_label='label'):

        # Split data from labels (don't want interaction with labels)
        X_data, _ = split_x_y(data, y_label=y_label)

        poly = PolynomialFeatures(**self.kwargs)
        self.fitted = poly.fit(X_data)
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.fitted is None:
            raise TransformError

        if y_label in data.columns:
            X_data, y_data = split_x_y(data, y_label=y_label)
        else:
            X_data = data
            y_data = None

        poly_data = self.fitted.transform(X_data)
        cols = self.fitted.get_feature_names(X_data.columns)
        cols = [c.replace(' ', '*') for c in cols]

        if self.append_data:
            return pd.concat((data, pd.DataFrame(poly_data, columns=cols)), axis=1)
        
        if y_data is not None:
            return pd.concat((y_data, pd.DataFrame(poly_data, columns=cols)), axis=1)

        return pd.DataFrame(poly_data, columns=cols)

################################################################################################
# SINE FEATURES
################################################################################################
# Takes the sine of every value in the given columns. If no columns given then the sine of
# every column is taken
 
class SinStep():
    def __init__(self, append_data=False, columns=None):
        self.description = "Sine"
        self.columns = columns
        self.append_data = append_data
        self.fitted = False
        self.removes_samples = False
    
    def fit(self, data, y_label='label'):
        self.fitted = True
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if not self.fitted:
            raise TransformError
        
        if self.columns is None:
            temp = data
        else:
            temp = data[self.columns]

        sin_data = np.sin(temp)
        new_cols = []
        for c in sin_data.columns:
            if y_label == c:
                new_cols.append(c)
            else:
                new_cols.append('sin_' + c)
        sin_data.columns = new_cols
        
        if self.append_data:   
            return pd.concat(data, sin_data)
        return sin_data