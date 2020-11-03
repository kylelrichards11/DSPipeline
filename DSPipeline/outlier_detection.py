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
        """ Uses angle based outlier detection to detect and remove outliers. Uses pyod’s ABOD class.
        
        Parameters
        ----------
        num_remove (int) : number of detected outliers to remove
        
        kwargs (dict, default={}) : arguments to pass to pyod's ABOD class initialization
        """
        self.description = 'Angle Based Outlier Detection'
        self.num_remove = num_remove
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = True

    def fit(self, X, y=None):
        """ Fits the outlier detection on the given data
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        abod = ABOD(**self.kwargs)
        self.fitted = abod.fit(X)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the given data using the previously fitted outlier detection method
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        if self.fitted is None:
            raise TransformError
        
        scores = pd.DataFrame(self.fitted.decision_scores_*-1, columns=['score'])
        scores = scores.sort_values('score', ascending=False)

        # Remove outliers
        rm_index = scores.head(self.num_remove).index
        X = X.drop(index=rm_index).reset_index(drop=True)

        if y is None:
            return X

        y = y.drop(index=rm_index).reset_index(drop=True)
        return X, y 

################################################################################################
# ISOLATION FOREST
################################################################################################
class IsoForestStep():
    def __init__(self, include_y=True, kwargs={'contamination': 'auto', 'behaviour': 'new'}):
        """ Uses an isolation forest to detect and remove outliers. Uses sklearn’s IsolationForest class. 
        
        Parameters
        ----------
        include_y (bool, default=True), Whether or not to include the y data when fitting the isolation forest

        kwargs (dict, default={'contamination': 'auto', 'behaviour': 'new'}) : arguments to pass to sklearn’s IsolationForest class initialization
        """
        self.description = "Isolation Forest Outlier Detection"
        self.include_y = include_y
        self.kwargs = kwargs
        self.changes_num_samples = True
        self.fitted = None

    def fit(self, X, y=None):
        """ Fits the outlier detection on the given data
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        self.fitted = IsolationForest(**self.kwargs)
        self.fitted.fit(X, y)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the given data using the previously fitted outlier detection method
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        if self.fitted is None:
            raise TransformError

        outlier_labels = self.fitted.predict(X)

        # Remove outliers from data
        for i in range(outlier_labels.shape[0]):
            if outlier_labels[i] == -1:
                X = X.drop(index=i)
                if y is not None:
                    y = y.drop(index=i)

        if y is None:
            return X.reset_index(drop=True)
        
        y = y.reset_index(drop=True)
        return X.reset_index(drop=True), y

################################################################################################
# LOCAL OUTLIER FACTOR
################################################################################################
class LOFStep():
    def __init__(self, include_y=True, kwargs={'contamination': 'auto'}):
        """ Uses the local outlier factor to detect and remove outliers. Uses sklearn’s LocalOutlierFactor class.
        
        Parameters
        ----------
        include_y (bool, default=True), Whether or not to include the y data when fitting the isolation forest

        kwargs (dict, default={'contamination': 'auto'}) : arguments to pass to sklearn’s IsolationForest class initialization
        """
        self.description = "Local Outlier Factor"
        self.include_y = include_y
        self.kwargs = kwargs
        self.fitted = None
        self.changes_num_samples = True

    def fit(self, X, y=None):
        """ Fits the outlier detection on the given data
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        self.fitted = LocalOutlierFactor(**self.kwargs)
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        """ Transforms the given data using the previously fitted outlier detection method
        
        Parameters
        ----------
        X (DataFrame) : training data

        y (DataFrame, default=None) : target values (if needed)

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        if self.fitted is None:
            raise TransformError

        outlier_labels = self.fitted.fit_predict(X, y)

        # Remove outliers from data
        for i in range(outlier_labels.shape[0]):
            if outlier_labels[i] == -1:
                X = X.drop(index=i)
                if y is not None:
                    y = y.drop(index=i)
        
        if y is None:
            return X.reset_index(drop=True)
        
        y = y.reset_index(drop=True)
        return X.reset_index(drop=True), y