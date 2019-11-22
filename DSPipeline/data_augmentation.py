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
        self.description = "ADASYN Data Augmentation"
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = True

    def fit(self, X, y=None):
        self.fitted = ADASYN(**self.kwargs)
        self.fitted.fit_resample(X, y)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.fitted is None:
            raise TransformError

        X_rs, y_rs = self.fitted.fit_resample(X, y)
        X_rs = pd.DataFrame(X_rs, columns=X.columns)
        y_rs = pd.Series(y_rs, name=y.name)
        if y is None:
            return X_rs
        return X_rs, y_rs

################################################################################################
# Synthetic Minority Over-Sampling Technique (SMOTE)
################################################################################################

class SMOTEStep():
    def __init__(self, smote_class=SMOTE, kwargs={}):
        self.description = "SMOTE Data Augmentation"
        self.smote_class = smote_class
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = True

    def fit(self, X, y=None):
        self.fitted = self.smote_class(**self.kwargs)
        self.fitted.fit_resample(X, y)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.fitted is None:
            raise TransformError

        X_rs, y_rs = self.fitted.fit_resample(X, y)
        X_rs = pd.DataFrame(X_rs, columns=X.columns)
        y_rs = pd.Series(y_rs, name=y.name)
        if y is None:
            return X_rs
        return X_rs, y_rs

