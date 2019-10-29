# External Imports

# Internal Imports

################################################################################################
# PIPELINE
################################################################################################
# Runs a pipeline with given steps

class Data_Pipeline():
    def __init__(self, steps):
        self.steps = steps

    def transform_train(self, data, y_label='label', verbose=False):
        for step in self.steps:
            if verbose:
                print(f'Transforming {step.description}')
            data = step.transform(data, y_label=y_label)
        return data

    def transform_test(self, data, y_label='label', verbose=False):
        for step in self.steps:
            if step.test_data:
                if verbose:
                    print(f'Transforming {step.description}')
                data = step.transform(data, y_label=y_label)
        return data

    def fit_transform(self, data, y_label='label', verbose=False):
        for step in self.steps:
            if verbose:
                print(f'Fit Transforming {step.description}')
            data = step.fit_transform(data, y_label=y_label)
        return data

################################################################################################
# An empty step to do nothing with the data
class Empty_Step_p():
    def __init__(self):
        self.description = "Empty Step"
        self.test_data = True

    def fit_transform(self, data, y_label='label'):
        return data

    def transform(self, data, y_label='label'):
        return data