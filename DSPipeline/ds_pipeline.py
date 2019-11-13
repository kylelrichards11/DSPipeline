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

    def fit(self, data, y_label='label', verbose=False):
        new_data = data
        for step in self.steps:
            if verbose:
                print(f'Fitting {step.description}')
            new_data = step.fit(new_data, y_label=y_label)
        if self.append_input:
            return pd.concat((data, new_data), axis=1).drop_duplicates(keep='last')
        return new_data

    def transform(self, data, y_label='label', allow_sample_removal=True, verbose=False):
        new_data = data
        for step in self.steps:
            if not allow_sample_removal and step.changes_num_samples:
                continue
            if verbose:
                print(f'Transforming {step.description}')
            new_data = step.transform(new_data, y_label=y_label)
        if self.append_input:
            return pd.concat((data, new_data)).drop_duplicates(keep='last')
        return new_data

    def fit_transform(self, data, y_label='label', allow_sample_removal=True, verbose=False):
        self.fit(data, y_label=y_label, verbose=verbose)
        return self.transform(data, y_label=y_label, allow_sample_removal=allow_sample_removal, verbose=verbose)

################################################################################################
# An empty step to do nothing with the data
class EmptyStep():
    def __init__(self, append_input=False):
        self.description = "Empty Step"
        self.changes_num_samples = False
        self.fitted = False

    def fit(self, data, y_label='label'):
        self.fitted = True
        return data

    def transform(self, data, y_label='label'):
        if self.fitted:
            return data
        raise TransformError