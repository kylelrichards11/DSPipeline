import pandas as pd
import numpy as np

from DSPipeline.errors import TransformError
from DSPipeline.ds_pipeline import Pipeline

# Create a new step that only selects features with the letter 'a'
class SelectAStep():
    def __init__(self):
        self.description = "Select features with \'a\'"
        self.changes_num_samples = False
        self.features = None

    def fit(self, X, y=None):
        features = [col for col in list(X.columns) if 'a' in col] 
        self.features = features
        return self.transform(X, y=y)

    def transform(self, X, y=None):
        if self.features is None:
            raise TransformError
        if y is None:
            return X[self.features]
        return X[self.features], y


if __name__ == "__main__":
    
    # Create Data
    shape = (5, 10)
    X = np.random.uniform(low=0, high=10, size=shape)
    cols = ['apple', 'banana', 'cucumber', 'date', 'eggplant', 'fennel', 'grape', 'honeydew', 'iceberg_lettuce', 'jalepeno']
    X = pd.DataFrame(X, columns=cols)

    # Create Pipeline
    pipeline = Pipeline([SelectAStep()])

    # Run Pipeline
    new_X = pipeline.fit_transform(X, verbose=True)
    print(new_X)