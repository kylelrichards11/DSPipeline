# External Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np

# Internal Imports
from .data_managing import split_x_y
from .errors import TransformError

################################################################################################
# SCALE DATA
################################################################################################

class StandardScalerStep():
    def __init__(self, append_input=False, kwargs={}):
        self.description = "Standard Scaler"
        self.append_input = append_input
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = False

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
        if self.append_input:
            new_cols = []
            for col in X_scaled.columns:
                new_cols.append(col + "_scaled")
            X_scaled.columns = new_cols
            X_scaled = pd.concat((data, X_scaled), axis=1)
        if y_data is not None:
            return pd.concat((X_scaled, y_data), axis=1)
        return X_scaled

################################################################################################
# PCA
################################################################################################
# Principal Component Analysis with or without appending principal components to original data

class PCAStep():
    def __init__(self, append_input=False, kwargs={}):
        self.description = 'PCA'
        self.kwargs = kwargs
        self.append_input = append_input
        self.fitted = None
        self.changes_num_samples = False

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
        if self.append_input:
            return pd.concat((data, pd.DataFrame(pca_data, columns=cols)), axis=1)
        if y_data is None:
            return pd.DataFrame(pca_data, columns=cols)
        return pd.concat((y_data, pd.DataFrame(pca_data, columns=cols)), axis=1)

################################################################################################
# POLYNOMIAL INTERACTIONS FEATURES
################################################################################################
# Generates interaction terms and polynomials with or without appending to original data

class PolyStep():

    def __init__(self, append_input=False, kwargs={}):
        self.description = 'Polynomial Features'
        self.kwargs = kwargs
        self.append_input = append_input
        self.fitted = None
        self.changes_num_samples = False

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

        if self.append_input:
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
    def __init__(self, append_input=False, columns=None, kwargs={}):
        self.description = "Sine"
        self.columns = columns
        self.append_input = append_input
        self.fitted = False
        self.changes_num_samples = False
        self.kwargs = kwargs
    
    def fit(self, data, y_label='label'):
        self.fitted = True
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if not self.fitted:
            raise TransformError
        
        if self.columns is None:
            temp = data.copy()
        else:
            temp = data[self.columns]

        if y_label in temp.columns:
            temp_X, temp_y = split_x_y(temp, y_label=y_label)
        else:
            temp_X = temp
            temp_y = None

        sin_data = np.sin(temp_X, **self.kwargs)
        new_cols = []
        for c in sin_data.columns:
            new_cols.append('sin_' + c)
        sin_data.columns = new_cols
        
        if self.append_input:   
            return pd.concat((data, sin_data), axis=1)

        if temp_y is not None:
            return pd.concat((sin_data, temp_y), axis=1)

        return sin_data

################################################################################################
# LOG FEATURES
################################################################################################
# Takes the log of every value in the given columns. If no columns given then the log of
# every column is taken

class LogStep():
    def __init__(self, append_input=False, columns=None, log_func=np.log, kwargs={}):
            self.description = "Log"
            self.columns = columns
            self.append_input = append_input
            self.fitted = False
            self.changes_num_samples = False
            self.log_func = log_func
        
    def fit(self, data, y_label='label'):
        self.fitted = True
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if not self.fitted:
            raise TransformError
        
        if self.columns is None:
            temp = data.copy()
        else:
            temp = data[self.columns]

        if y_label in temp.columns:
            temp_X, temp_y = split_x_y(temp, y_label=y_label)
        else:
            temp_X = temp
            temp_y = None

        log_data = self.log_func(temp)
        new_cols = []
        for c in log_data.columns:
            new_cols.append('log_' + c)
        log_data.columns = new_cols
        
        if self.append_input:   
            return pd.concat((data, log_data), axis=1)

        if temp_y is not None:
            return pd.concat((log_data, temp_y), axis=1)

        return log_data

################################################################################################
# LDA TRANSFORMATION
################################################################################################
# Uses linear discriminant analysis to project the data into the most seperable components

class LDATransformStep():

    def __init__(self, append_input=False, kwargs={}):
        self.description = "LDA Feature Transformation"
        self.append_input = append_input
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = False

    def fit(self, data, y_label='label'):
        X_data, y_data = split_x_y(data, y_label=y_label)
        lda = LinearDiscriminantAnalysis(**self.kwargs)
        self.fitted = lda.fit(X_data, y_data)
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.fitted is None:
            raise TransformError

        if y_label in data.columns:
            X_data, y_data = split_x_y(data, y_label=y_label)
        else:
            X_data = data
            y_data = None

        lda_data = self.fitted.transform(X_data)
        
        lda_cols = []
        for i in range(1, lda_data.shape[1]+1):
            lda_cols.append(f"LDA_{i}")

        lda_data = pd.DataFrame(lda_data, columns=lda_cols)

        if self.append_input:
            return pd.concat((data, lda_data), axis=1)
        
        if y_data is None:
            return lda_data

        return pd.concat((lda_data, y_data), axis=1)