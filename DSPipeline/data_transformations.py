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

    def fit(self, X, y=None):
        scaler = StandardScaler(**self.kwargs)  
        self.fitted = scaler.fit(X)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.fitted is None:
            raise TransformError

        X_scaled = pd.DataFrame(self.fitted.transform(X), columns=X.columns)
        if self.append_input:
            new_cols = []
            for col in X_scaled.columns:
                new_cols.append(col + "_scaled")
            X_scaled.columns = new_cols
            X_scaled = pd.concat((X, X_scaled), axis=1)
        
        if y is None:
            return X_scaled
        return X_scaled, y

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

    def fit(self, X, y=None):
        pca_model = PCA(**self.kwargs)
        self.fitted = pca_model.fit(X)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.fitted is None:
            raise TransformError

        pca_data = self.fitted.transform(X)
        
        # Get column names for post pca dataframe
        cols = []
        for i in range(1, pca_data.shape[1]+1):
            cols.append("PC_" + str(i))

        # Return pca data with or without appending
        if self.append_input:
            if y is None:
                return pd.concat((X, pd.DataFrame(pca_data, columns=cols)), axis=1)
            return pd.concat((X, pd.DataFrame(pca_data, columns=cols)), axis=1), y
        if y is None:
            return pd.DataFrame(pca_data, columns=cols)
        return pd.DataFrame(pca_data, columns=cols), y

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

    def fit(self, X, y=None):
        poly = PolynomialFeatures(**self.kwargs)
        self.fitted = poly.fit(X)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.fitted is None:
            raise TransformError

        poly_data = self.fitted.transform(X)
        cols = self.fitted.get_feature_names(X.columns)
        cols = [c.replace(' ', '*') for c in cols]

        if self.append_input:
            if y is None:
                return pd.concat((X, pd.DataFrame(poly_data, columns=cols)), axis=1)
            return pd.concat((X, pd.DataFrame(poly_data, columns=cols)), axis=1), y

        if y is None:
            return pd.DataFrame(poly_data, columns=cols)
        return pd.DataFrame(poly_data, columns=cols), y

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
    
    def fit(self, X, y=None):
        self.fitted = True
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if not self.fitted:
            raise TransformError
        
        if self.columns is None:
            temp_X = X.copy()
        else:
            temp_X = X[self.columns]

        sin_data = np.sin(temp_X, **self.kwargs)
        new_cols = []
        for c in sin_data.columns:
            new_cols.append('sin_' + c)
        sin_data.columns = new_cols
        
        if self.append_input:
            if y is None:
                return pd.concat((X, sin_data), axis=1)
            return pd.concat((X, sin_data), axis=1), y

        if y is None:
            return sin_data
        return sin_data, y

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
        
    def fit(self, X, y=None):
        self.fitted = True
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if not self.fitted:
            raise TransformError
        
        if self.columns is None:
            temp_X = X.copy()
        else:
            temp_X = X[self.columns]

        log_data = self.log_func(temp_X)
        new_cols = []
        for c in log_data.columns:
            new_cols.append('log_' + c)
        log_data.columns = new_cols
        
        if self.append_input:   
            if y is None:
                return pd.concat((X, log_data), axis=1)
            return pd.concat((X, log_data), axis=1), y

        if y is None:
            return log_data
        return log_data, y

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

    def fit(self, X, y=None):
        lda = LinearDiscriminantAnalysis(**self.kwargs)
        self.fitted = lda.fit(X, y)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.fitted is None:
            raise TransformError

        lda_data = self.fitted.transform(X)
        
        lda_cols = []
        for i in range(1, lda_data.shape[1]+1):
            lda_cols.append(f"LDA_{i}")

        lda_data = pd.DataFrame(lda_data, columns=lda_cols)

        if self.append_input:
            if y is None:
                return pd.concat((X, lda_data), axis=1)
            return pd.concat((X, lda_data), axis=1), y

        if y is None:
            return lda_data
        return lda_data, y