# External Imports

# Internal Imports

################################################################################################
# PIPELINE
################################################################################################
# Runs a pipeline with given steps

class DS_Pipeline():
    def __init__(self, steps):
        self.steps = steps

    def transform(self, data, y_label='label', allow_sample_removal=True, verbose=False):
        for step in self.steps:
            if not allow_sample_removal and step.removes_samples:
                continue
            if verbose:
                print(f'Transforming {step.description}')
            data = step.transform(data, y_label=y_label)
        return data

    def fit(self, data, y_label='label', verbose=False):
        for step in self.steps:
            if verbose:
                print(f'Fitting {step.description}')
            data = step.fit(data, y_label=y_label)

    def fit_transform(self, data, y_label='label', allow_sample_removal=True, verbose=False):
        self.fit(data, y_label=y_label, verbose=verbose)
        return self.transform(data, y_label=y_label, allow_sample_removal=allow_sample_removal, verbose=verbose)

################################################################################################
# An empty step to do nothing with the data
class Empty_Step():
    def __init__(self):
        self.description = "Empty Step"
        self.test_data = True

    def fit(self, data, y_label='label'):
        return data

    def transform(self, data, y_label='label'):
        return data