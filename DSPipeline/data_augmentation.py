# External Imports
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE

# Internal Imports
from .data_managing import split_x_y
from .errors import TransformError

################################################################################################
# Adaptive Synthetic Over-Sampling Technique (ADASYN)
################################################################################################
class ADASYNStep():
    def __init__(self, kwargs={}):
        """ Adaptive Synthetic Over-Sampling Technique (ADASYN) to create balanced samples. Uses imblearn’s ADASYN class. 
        
        Parameters
        ----------
        kwargs (dict, default={}) : arguments to pass to imblearn ADASYN class
        
        """
        self.description = "ADASYN Data Augmentation"
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = True

    def fit(self, X, y=None):
        """ Fits the ADASYN to given data
        
        Parameters
        ----------
        X (DataFrame) : the data to fit

        y (DataFrame) : the labels for X

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        if y is None:
            print(f"{self.description} step is supervised and needs target values")
            raise ValueError
        self.fitted = ADASYN(**self.kwargs)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the given data using previously fitted ADASYN
        
        Parameters
        ----------
        X (DataFrame) : the data to fit

        y (DataFrame) : the labels for X

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        if self.fitted is None:
            raise TransformError

        X_rs, y_rs = self.fitted.fit_resample(X, y)
        X_rs = pd.DataFrame(X_rs, columns=X.columns)
        y_rs = pd.Series(y_rs, name=y.name)
        return X_rs, y_rs

################################################################################################
# Synthetic Minority Over-Sampling Technique (SMOTE)
################################################################################################
class SMOTEStep():
    def __init__(self, smote_class=SMOTE, kwargs={}):
        """ Uses Synthetic Minority Over-Sampling Technique (SMOTE) to create balanced samples. Uses imblearn’s SMOTE family of classes.
        
        Parameters
        ----------
        smote_class (object, default=SMOTE) : the smote class to use for the data augmentation. imblearn offers different classes such as SVMSMOTE, KMeansSMOTE, etc.

        kwargs (dict, default={}) : arguments to pass to the smote_class upon initialization
        """
        self.description = "SMOTE Data Augmentation"
        self.smote_class = smote_class
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = True

    def fit(self, X, y=None):
        """ Fits the smote_class to given data
        
        Parameters
        ----------
        X (DataFrame) : the data to fit

        y (DataFrame) : the labels for X

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        self.fitted = self.smote_class(**self.kwargs)
        if y is None:
            print(f"{self.description} step is supervised and needs target values")
            raise ValueError
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the given data using previously fitted smote_class
        
        Parameters
        ----------
        X (DataFrame) : the data to fit

        y (DataFrame) : the labels for X

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        if self.fitted is None:
            raise TransformError

        X_rs, y_rs = self.fitted.fit_resample(X, y)
        X_rs = pd.DataFrame(X_rs, columns=X.columns)
        y_rs = pd.Series(y_rs, name=y.name)
        return X_rs, y_rs
