# External Imports
import pandas as pd

# Internal Imports
from DSPipeline.errors import TransformError

################################################################################################
# PIPELINE
################################################################################################
class Pipeline():
    def __init__(self, steps, append_input=False):
        """ This class stores all of the steps that can be applied to data. It can also be used as a single step containing other sub steps.
        
        Parameters
        ----------
        steps (list) : a list of step objects to use in the pipeline in the given order

        append_input (bool, default=False) : Whether to append the transformed data to the given data, or to only keep the transformed data
        """
        self.steps = steps
        self.append_input = append_input
        self.description = f"Pipeline Step with {[s.description for s in steps]}"
        self.changes_num_samples = False

    def fit(self, X, y=None, verbose=False):
        """ Fits the pipeline on the given data
        
        Parameters
        ----------
        X (DataFrame) : the training data

        y (DataFrame, default=None) : target values (if needed)

        verbose (bool, default=False) : whether or not to output progress of fitting the pipeline

        Returns
        -------
        (DataFrame, DataFrame) : a tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        new_X = X.copy()
        for step in self.steps:
            if verbose:
                print(f'Fitting {step.description}')
            if y is None:
                new_X = step.fit(new_X)
            else:
                new_X, y = step.fit(new_X, y=y)
        if self.append_input:
            if y is None:
                return pd.concat((X, new_X), axis=1)
            return pd.concat((X, new_X), axis=1), y
        if y is None:
            return new_X
        return new_X, y

    def transform(self, X, y=None, allow_sample_removal=True, verbose=False):
        """ Transforms the given data using the previously fitted pipeline
        
        Parameters
        ----------
        X (DataFrame) : The training data

        y (DataFrame, default=None) : Target values (if needed)

        allow_sample_removal (bool, default=True) : Whether or not the pipeline is allowed to remove any samples from the data frame. Generally this is okay in train data and not okay in test data.

        verbose (bool, default=False) : Whether or not to output progress of fitting the pipeline

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        new_X = X.copy()
        for step in self.steps:
            if not allow_sample_removal and step.changes_num_samples:
                continue
            if verbose:
                print(f'Transforming {step.description}')
            if y is None:
                new_X = step.transform(new_X)
            else:
                new_X, y = step.transform(new_X, y=y)
        if self.append_input:
            if y is None:
                return pd.concat((X, new_X), axis=1)
            return pd.concat((X, new_X), axis=1), y
        if y is None:
            return new_X
        return new_X, y

    def fit_transform(self, X, y=None, allow_sample_removal=True, verbose=False):
        """ Fits the pipeline and then transforms the given data
        
        Parameters
        ----------
        X (DataFrame) : The training data

        y (DataFrame, default=None) : Target values (if needed)

        allow_sample_removal (bool, default=True) : Whether or not the pipeline is allowed to remove any samples from the data frame. Generally this is okay in train data and not okay in test data.

        verbose (bool, default=False) : Whether or not to output progress of fitting the pipeline

        Returns
        -------
        (DataFrame, DataFrame) : A tuple of the transformed DataFrames, the first being the X data and the second being the y data
        """
        self.fit(X, y=y, verbose=verbose)
        return self.transform(X, y=y, allow_sample_removal=allow_sample_removal, verbose=verbose)

################################################################################################
# EMPTY STEP
################################################################################################
class EmptyStep():
    def __init__(self):
        """ A pipeline step that does nothing to the data. It is useful as a placeholder.
        
        Parameters
        ----------
        None
        """
        self.description = "Empty Step"
        self.changes_num_samples = False
        self.fitted = False

    def fit(self, X, y=None):
        """ Placeholder fit method """
        self.fitted = True
        if y is None:
            return X
        return X, y

    def transform(self, X, y=None):
        """ Placeholder transform method """
        if self.fitted:
            if y is None:
                return X
            return X, y
        raise TransformError