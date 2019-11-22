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
        self.changes_num_samples = False
        self.fitted = False

    def fit(self, X, y=None):
        self.fitted = True
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if not self.fitted:
            raise TransformError

        # Try to keep features if they are there
        try:
            if y is None:
                return X[self.features]
            return X[self.features], y
        except KeyError:
            print("Could not fit features:")
            print(self.features)
            print("to data with features:")
            print(list(X.columns))
            raise KeyError

################################################################################################
## REGRESSION FOREST FEATURE SELECTION
################################################################################################

class TreeSelectionStep():
    def __init__(self, tree_model=ExtraTreesRegressor, tree_kwargs={'n_estimators':100}, select_kwargs={}):
        self.description = 'Tree Feature Selection'
        self.tree_model = tree_model
        self.tree_kwargs = tree_kwargs
        self.select_kwargs = select_kwargs
        self.changes_num_samples = False
        self.features = None

    def fit(self, X, y=None):
        model = self.tree_model(**self.tree_kwargs)
        fitter = SelectFromModel(model, **self.select_kwargs)

        cols = X.columns
        fitter.fit(X, y)

        features_i = fitter.get_support(indices=True)
        feature_names = []
        for i in features_i:
            feature_names.append(cols[i])
        self.features = feature_names
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.features is None:
            raise TransformError

        X = X.loc[:, X.columns.isin(self.features)]
        if y is None:
            return X
        return X, y

################################################################################################
# PEARSON CORRELATION FEATURE SELECTION
################################################################################################

class PearsonCorrStep():
    def __init__(self, num_features, kwargs={}):
        self.description = "Pearson Correlation Feature Selection"
        self.num_features = num_features
        self.kwargs = kwargs
        self.features = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        # if type(X) != type(pd.DataFrame()):
        #     y_label = 'y_column'
        #     X = pd.DataFrame(X)
        #     y = pd.DataFrame(y, columns=[y_label])
        if y is not None:
            y_label = y.name
        data = pd.concat((X, y), axis=1)
        corr = data.corr(**self.kwargs)
        corr_target = abs(corr[y_label])
        if self.num_features < 1:
            relevant_features = corr_target[corr_target > self.num_features]
        else:
            corr_target = corr_target.sort_values(ascending=False)
            relevant_features = corr_target.iloc[:(self.num_features + 1)]
        self.features = relevant_features.drop(index=y_label)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.features is None:
            raise TransformError
        if y is None:
            return X.loc[:, X.columns.isin(self.features.index)]
        return X.loc[:, X.columns.isin(self.features.index)], y

################################################################################################
# CHI SQUARED FEATURE SELECTION
################################################################################################

class ChiSqSelectionStep():
    def __init__(self, select_kwargs={}):
        self.description = "Chi Squared Feature Selection"
        self.select_kwargs = select_kwargs
        self.features = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
        chi_selector = SelectKBest(chi2, **self.select_kwargs)
        chi_selector.fit(X_norm, y)
        chi_support = chi_selector.get_support()
        self.features = X.loc[:, chi_support].columns.tolist()
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.features is None:
            raise TransformError
        if y is None:
            return X.loc[:, X.columns.isin(self.features)]
        return X.loc[:, X.columns.isin(self.features)], y

################################################################################################
# LASSO FEATURE SELECTION
################################################################################################

class LassoSelectionStep():
    def __init__(self, lasso_kwargs={}, select_kwargs={}):
        self.description = "Lasso Feature Selection"
        self.lasso_kwargs = lasso_kwargs
        self.select_kwargs = select_kwargs
        self.features = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        embeded_lr_selector = SelectFromModel(Lasso(**self.lasso_kwargs), **self.select_kwargs)
        embeded_lr_selector.fit(X, y)

        embeded_lr_support = embeded_lr_selector.get_support()
        self.features = X.loc[:, embeded_lr_support].columns.tolist()
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.features is None:
            raise TransformError
        
        if y is None:
            return X.loc[:, X.columns.isin(self.features)]
        return X.loc[:, X.columns.isin(self.features)], y
