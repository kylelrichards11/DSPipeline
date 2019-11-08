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

    def fit(self, data, y_label='label'):
        self.fitted = ADASYN(**self.kwargs)
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.fitted is None:
            raise TransformError

        X_data, y_data = split_x_y(data, y_label=y_label)
        X_rs, y_rs = self.fitted.fit_resample(X_data, y_data)
        X_rs = pd.DataFrame(X_rs, columns=X_data.columns)
        y_rs = pd.DataFrame(y_rs, columns=[y_data.name])
        return pd.concat((X_rs, y_rs), axis=1)

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

    def fit(self, data, y_label='label'):
        self.fitted = self.smote_class(**self.kwargs)
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.fitted is None:
            raise TransformError

        X_data, y_data = split_x_y(data, y_label=y_label)
        X_rs, y_rs = self.fitted.fit_resample(X_data, y_data)
        X_rs = pd.DataFrame(X_rs, columns=X_data.columns)
        y_rs = pd.DataFrame(y_rs, columns=[y_data.name])
        return pd.concat((X_rs, y_rs), axis=1)

