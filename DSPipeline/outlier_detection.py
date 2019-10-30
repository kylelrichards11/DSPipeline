# External Imports
import pandas as pd
from pyod.models.abod import ABOD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Internal Imports
from .data_managing import split_x_y
from .errors import TransformError

################################################################################################
# ANGLE BASED OUTLIER DETECTION
################################################################################################

class ABODStep():
    def __init__(self, num_remove, kwargs={}):
        self.description = 'Angle Based Outlier Detection'
        self.num_remove = num_remove
        self.kwargs = kwargs
        self.fitted = None
        self.removes_samples = True

    def fit(self, data, y_label='label'):
        abod = ABOD(**self.kwargs)
        self.fitted = abod.fit(data)
        return data

    def transform(self, data, y_label='label'):
        scores = pd.DataFrame(self.fitted.decision_scores_*-1, columns=['score'])
        scores = scores.sort_values('score', ascending=False)

        # Remove outliers
        rm_index = scores.head(self.num_remove).index
        data = data.drop(index=rm_index)

        return data.reset_index(drop=True)

################################################################################################
# ISOLATION FOREST
################################################################################################

class IsoForestStep():
    def __init__(self, include_y=True, kwargs={'contamination': 'auto', 'behaviour': 'new'}):
        self.description = "Isolation Forest Outlier Detection"
        self.include_y = include_y
        self.kwargs = kwargs
        self.removes_samples = True
        self.fitted = None

    def fit(self, data, y_label='label'):
        iso = IsolationForest(**self.kwargs)
        if self.include_y:
            self.fitted = iso.fit(data)
        else:
            X_data, _ = split_x_y(data, y_label=y_label)
            self.fitted = iso.fit(X_data)
        return data

    def transform(self, data, y_label='label'):
        if self.fitted is None:
            raise TransformError

        if self.include_y:
            outlier_labels = self.fitted.predict(data)
        else:
            X_data, _ = split_x_y(data, y_label=y_label)
            outlier_labels = self.fitted.predict(X_data)

        # Remove outliers from data
        for i in range(outlier_labels.shape[0]):
            if outlier_labels[i] == -1:
                data = data.drop(index=i)
        
        return data.reset_index(drop=True)

################################################################################################
# LOCAL OUTLIER FACTOR
################################################################################################

class LOFStep():
    def __init__(self, include_y=True, kwargs={'contamination': 'auto'}):
        self.description = "Local Outlier Factor"
        self.include_y = include_y
        self.kwargs = kwargs
        self.fitted = None
        self.removes_samples = True

    def fit(self, data, y_label='label'):
        self.fitted = LocalOutlierFactor(**self.kwargs)
        return data

    def transform(self, data, y_label='label'):
        if self.fitted is None:
            raise TransformError

        if self.include_y:
            outlier_labels = self.fitted.fit_predict(data)
        else:
            X_data, _ = split_x_y(data, y_label=y_label)
            outlier_labels = self.fitted.fit_predict(X_data)

        # Remove outliers from data
        for i in range(outlier_labels.shape[0]):
            if outlier_labels[i] == -1:
                data = data.drop(index=i)
        
        return data.reset_index(drop=True)