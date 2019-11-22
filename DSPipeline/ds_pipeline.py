# External Imports
import pandas as pd

# Internal Imports
from DSPipeline.errors import TransformError

################################################################################################
# PIPELINE
################################################################################################
# Runs a pipeline with given steps

class Pipeline():
    def __init__(self, steps, append_input=False):
        self.steps = steps
        self.append_input = append_input
        self.description = f"Pipeline Step with {[s.description for s in steps]}"
        self.changes_num_samples = False

    def fit(self, X, y=None, verbose=False):
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
        self.fit(X, y=y, verbose=verbose)
        return self.transform(X, y=y, allow_sample_removal=allow_sample_removal, verbose=verbose)

################################################################################################
# An empty step to do nothing with the data
class EmptyStep():
    def __init__(self):
        self.description = "Empty Step"
        self.changes_num_samples = False
        self.fitted = False

    def fit(self, X, y=None):
        self.fitted = True
        if y is None:
            return X
        return X, y

    def transform(self, X, y=None):
        if self.fitted:
            if y is None:
                return X
            return X, y
        raise TransformError