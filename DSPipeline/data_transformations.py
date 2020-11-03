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
        """ Scales the data with sklearn's StandardScaler 
        
        Parameters
        ----------
        append_input (bool, default=False) : Whether to append the scaled features to the given data, or to only keep the transformed data

        kwargs (dict, default={}) : Arguments to be passed to sklearn’s StandardScaler class
        """
        self.description = "Standard Scaler"
        self.append_input = append_input
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        """ Fits the standard scaler 
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        scaler = StandardScaler(**self.kwargs)  
        self.fitted = scaler.fit(X)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the input data using the previously fitted step 
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
class PCAStep():
    def __init__(self, append_input=False, kwargs={}):
        """ Applies principal component analysis to the given data with sklearn’s PCA.
        
        Parameters
        ----------
        append_input (bool, default=False) : Whether to append the scaled features to the given data, or to only keep the transformed data

        kwargs (dict, default={}) : Arguments to be passed to sklearn’s PCA class
        """
        self.description = 'PCA'
        self.kwargs = kwargs
        self.append_input = append_input
        self.fitted = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        """ Fits PCA
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        pca_model = PCA(**self.kwargs)
        self.fitted = pca_model.fit(X)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the input data using the previously fitted step 
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
class PolyStep():

    def __init__(self, append_input=False, kwargs={}):
        """ Applies polynomial feature combinations to the given data with sklearn’s PolynomialFeatures
        
        Parameters
        ----------
        append_input (bool, default=False) : Whether to append the scaled features to the given data, or to only keep the transformed data

        kwargs (dict, default={}) : Arguments to be passed to sklearn’s PolynomialFeatures class
        """
        self.description = 'Polynomial Features'
        self.kwargs = kwargs
        self.append_input = append_input
        self.fitted = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        """ Fits the polynomial features
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        poly = PolynomialFeatures(**self.kwargs)
        self.fitted = poly.fit(X)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the input data using the previously fitted step 
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
class SinStep():
    def __init__(self, append_input=False, columns=None, kwargs={}):
        """ Applies the sine function to specified columns of the given data. It uses numpy’s sine function.
        
        Parameters
        ----------
        append_input (bool, default=False) : Whether to append the scaled features to the given data, or to only keep the transformed data

        columns (object, default=None) : The columns to apply the sine function to. If None all columns are used.

        kwargs (dict, default={}) : Arguments to be passed to numpy's sine function
        """
        self.description = "Sine"
        self.columns = columns
        self.append_input = append_input
        self.fitted = False
        self.changes_num_samples = False
        self.kwargs = kwargs
    
    def fit(self, X, y=None):
        """ Marks the object as having been fitted. As this is just a sine tranformation, no fitting is expressly needed, but for consistency it is required to call fit before transforming.
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        self.fitted = True
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the input data using the previously fitted step 
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
class LogStep():
    def __init__(self, append_input=False, columns=None, log_func=np.log, kwargs={}):
        """  Applies a logarithmic function to specified columns of the given data. The default uses numpy’s log function.
        
        Parameters
        ----------
        append_input (bool, default=False) : Whether to append the scaled features to the given data, or to only keep the transformed data

        columns (object, default=None) : The columns to apply the sine function to. If None all columns are used.

        log_func (function, default=np.log) : The type of log function to use; np.log, np.log10, etc.

        kwargs (dict, default={}) : Arguments to be passed to the chosen log_func
        """
        self.description = "Log"
        self.columns = columns
        self.append_input = append_input
        self.fitted = False
        self.changes_num_samples = False
        self.log_func = log_func
        self.kwargs = kwargs
        
    def fit(self, X, y=None):
        """ Marks the object as having been fitted. As this is just a log tranformation, no fitting is expressly needed, but for consistency it is required to call fit before transforming.
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        self.fitted = True
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the input data using the previously fitted step 
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        if not self.fitted:
            raise TransformError
        
        if self.columns is None:
            temp_X = X.copy()
        else:
            temp_X = X[self.columns]

        log_data = self.log_func(temp_X, **self.kwargs)
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
class LDATransformStep():

    def __init__(self, append_input=False, kwargs={}):
        """ Applies linear discriminant analysis to project the data into the most seperable components with sklearn’s LinearDiscriminantAnalysis
        
        Parameters
        ----------
        append_input (bool, default=False) : Whether to append the scaled features to the given data, or to only keep the transformed data

        kwargs (dict, default={}) : Arguments to be passed to sklearn’s LinearDiscriminantAnalysis class
        """
        self.description = "LDA Feature Transformation"
        self.append_input = append_input
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        """ Fits the LDA to the training data
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        lda = LinearDiscriminantAnalysis(**self.kwargs)
        self.fitted = lda.fit(X, y)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the input data using the previously fitted step 
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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