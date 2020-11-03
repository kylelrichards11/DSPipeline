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
class ListSelectionStep():
    def __init__(self, features):
        """ Selects columns based on a given list of columns to keep.
        
        Parameters
        ----------
        features (list) : List of columns to keep. The y_label column is always kept.
        """
        self.description = f'Select features: {str(features)}'
        self.features = features
        self.changes_num_samples = False
        self.fitted = False

    def fit(self, X, y=None):
        """ Sets the step as fitted. Does not actually fit anything since this feature selection is not specific to the data
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        self.fitted = True
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the given data using the previously fitted selection
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
## TREE SELECTION
################################################################################################
class TreeSelectionStep():
    def __init__(self, tree_model=ExtraTreesRegressor, tree_kwargs={'n_estimators':100}, select_kwargs={}):
        """Uses a tree to select features. Uses sklearn’s ExtraTreesRegressor (default) and SelectFromModel classes.
        
        Parameters
        ----------
        tree_model (object) : the type of model to use as a tree. See https://scikit-learn.org/stable/modules/ensemble.html#ensemble for more

        tree_kwargs (dict, default={}) : arguments to pass to the tree model initialization

        select_kwargs (dict, default={}) : arguments to pass to sklearn's SelectFromModel class's initializiation
        """
        self.description = 'Tree Feature Selection'
        self.tree_model = tree_model
        self.tree_kwargs = tree_kwargs
        self.select_kwargs = select_kwargs
        self.changes_num_samples = False
        self.features = None

    def fit(self, X, y=None):
        """ Fits the selection on the given data
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
        """ Transforms the given data using the previously fitted selection
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
        """ Uses pearson’s correlation to select features. Uses pandas’s corr method. 
        
        Parameters
        ----------
        num_features (float) : Number of features to keep. If less than 1, then that is the minimum correlation value

        kwargs (dict, default={}) : Arguments to pass to panda's corr method.
        """
        self.description = "Pearson Correlation Feature Selection"
        self.num_features = num_features
        self.kwargs = kwargs
        self.features = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        """ Fits the selection on the given data
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
        """ Transforms the given data using the previously fitted selection
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
        """ Uses the chi squared test to select relevant features for classification tasks. Uses sklearn’s chi2 and SelectKBest classes.

        Parameters
        ----------
        select_kwargs (dict, default={}) : arguments to pass to sklearn's SelectKBest class initialization
        """
        self.description = "Chi Squared Feature Selection"
        self.select_kwargs = select_kwargs
        self.features = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        """ Fits the selection on the given data
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
        chi_selector = SelectKBest(chi2, **self.select_kwargs)
        chi_selector.fit(X_norm, y)
        chi_support = chi_selector.get_support()
        self.features = X.loc[:, chi_support].columns.tolist()
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the given data using the previously fitted selection
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
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
        """ Uses lasso regularization to select features. Uses sklearn’s Lasso and SelectKBest classes. 
        
        Parameters
        ----------
        lasso_kwargs (dict, default={}) : arguments to pass to sklearn's Lasso class initializiation
        
        select_kwargs (dict, default={}) : arguments to pass to sklearn's SelectKBest class initializiation
        """
        self.description = "Lasso Feature Selection"
        self.lasso_kwargs = lasso_kwargs
        self.select_kwargs = select_kwargs
        self.features = None
        self.changes_num_samples = False

    def fit(self, X, y=None):
        """ Fits the selection on the given data
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        embeded_lr_selector = SelectFromModel(Lasso(**self.lasso_kwargs), **self.select_kwargs)
        embeded_lr_selector.fit(X, y)

        embeded_lr_support = embeded_lr_selector.get_support()
        self.features = X.loc[:, embeded_lr_support].columns.tolist()
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the given data using the previously fitted selection
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        if self.features is None:
            raise TransformError
        
        if y is None:
            return X.loc[:, X.columns.isin(self.features)]
        return X.loc[:, X.columns.isin(self.features)], y
