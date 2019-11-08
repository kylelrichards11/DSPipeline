# External Imports
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso

# Internal Imports
from .data_managing import split_x_y
from .errors import TransformError

################################################################################################
## LIST FEATURE SELECTION
################################################################################################
# Selects based on the given list of features

class ListSelectionStep():
    def __init__(self, features):
        self.description = 'Select features: ' + str(features)
        self.features = features
        self.removes_samples = False
        self.fitted = False

    def fit(self, data, y_label='label'):
        self.fitted = True
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if not self.fitted:
            raise TransformError

        # Make sure we do not drop y
        features = self.features.copy()
        if (y_label in data.columns) and (y_label not in self.features):
            features.append(y_label)

        # Try to keep features if they are there
        try:
            return data[features]
        except KeyError:
            print("Could not fit features:")
            print(self.features)
            print("to data with features:")
            print(list(data.columns))
            raise KeyError

################################################################################################
## REGRESSION FOREST FEATURE SELECTION
################################################################################################

class RegTreeSelectionStep():
    def __init__(self, tree_kwargs={'n_estimators':100}, select_kwargs={}):
        self.description = 'Regression Tree Forest Feature Selection'
        self.tree_kwargs = tree_kwargs
        self.select_kwargs = select_kwargs
        self.removes_samples = False
        self.features = None

    def fit(self, data, y_label='label'):
        reg_forest = ExtraTreesRegressor(**self.tree_kwargs)
        reg_forest_fitter = SelectFromModel(reg_forest, **self.select_kwargs)

        cols = data.columns
        X_data, y_data = split_x_y(data, y_label=y_label)

        reg_forest_fitter.fit(X_data, y_data)

        features_i = reg_forest_fitter.get_support(indices=True)
        feature_names = []
        for i in features_i:
            feature_names.append(cols[i])
        self.features = feature_names
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.features is None:
            raise TransformError

        if y_label in data.columns:
            X_data, y_data = split_x_y(data, y_label=y_label)
        else:
            X_data = data
            y_data = None

        X_data = data.loc[:, data.columns.isin(self.features)]
        if y_data is None:
            return X_data
        return pd.concat((X_data, y_data), axis=1)

################################################################################################
# PEARSON CORRELATION FEATURE SELECTION
################################################################################################

class PearsonCorrStep():
    def __init__(self, threshold, kwargs={}):
        self.description = "Pearson Correlation Feature Selection"
        self.threshold = threshold
        self.kwargs = kwargs
        self.features = None
        self.removes_samples = False

    def fit(self, data, y_label='label'):
        corr = data.corr(**self.kwargs)
        corr_target = abs(corr[y_label])
        relevant_features = corr_target[corr_target > self.threshold]
        self.features = relevant_features.drop(index=y_label)
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.features is None:
            raise TransformError

        if y_label in data.columns:
            f = list(self.features.index)
            f.append(y_label)
            return data.loc[:, data.columns.isin(f)]
        return data.loc[: , data.columns.isin(self.features.index)]

################################################################################################
# CHI SQUARED FEATURE SELECTION
################################################################################################

class ChiSqSelectionStep():
    def __init__(self, select_kwargs={}):
        self.description = "Chi Squared Feature Selection"
        self.select_kwargs = select_kwargs
        self.features = None
        self.removes_samples = False

    def fit(self, data, y_label='label'):
        X_data, y_data = split_x_y(data, y_label=y_label)
        X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_data), columns=X_data.columns)
        chi_selector = SelectKBest(chi2, **self.select_kwargs)
        chi_selector.fit(X_norm, y_data)
        chi_support = chi_selector.get_support()
        self.features = X_data.loc[:, chi_support].columns.tolist()
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.features is None:
            raise TransformError

        features = self.features.copy()
        if y_label in data.columns:
            features.append(y_label)
            return data.loc[:, data.columns.isin(features)]
        return data.loc[:, data.columns.isin(features)]

################################################################################################
# LASSO FEATURE SELECTION
################################################################################################

class LassoSelectionStep():
    def __init__(self, lasso_kwargs={}, select_kwargs={}):
        self.description = "Lasso Feature Selection"
        self.lasso_kwargs = lasso_kwargs
        self.select_kwargs = select_kwargs
        self.features = None
        self.removes_samples = False

    def fit(self, data, y_label='label'):
        X_data, y_data = split_x_y(data, y_label=y_label)

        embeded_lr_selector = SelectFromModel(Lasso(**self.lasso_kwargs), **self.select_kwargs)
        embeded_lr_selector.fit(X_data, y_data)

        embeded_lr_support = embeded_lr_selector.get_support()
        self.features = X_data.loc[:, embeded_lr_support].columns.tolist()
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='label'):
        if self.features is None:
            raise TransformError
        
        if y_label in data.columns:
            self.features.append(y_label)
            return data.loc[:, data.columns.isin(self.features)]
        return data.loc[:, data.columns.isin(self.features)]
