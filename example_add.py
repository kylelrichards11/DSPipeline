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

    def fit(self, data, y_label='y'):
        features = [col for col in list(data.columns) if 'a' in col] 
        # Make sure we don't drop the label!
        if y_label not in features:
            features.append(y_label)
        self.features = features
        return self.transform(data, y_label=y_label)

    def transform(self, data, y_label='y'):
        if self.features is None:
            raise TransformError
        return data[self.features]

# MAIN
if __name__ == "__main__":
    
    # Create Data
    shape = (5, 10)
    data = np.random.uniform(low=0, high=10, size=shape)
    cols = ['apple', 'banana', 'cucumber', 'date', 'eggplant', 'fennel', 'grape', 'honeydew', 'iceberg_lettuce', 'y']
    data = pd.DataFrame(data, columns=cols)

    # Create Pipeline
    pipeline = Pipeline([SelectAStep()])

    # Run Pipeline
    new_data = pipeline.fit_transform(data, y_label='y', verbose=True)
    print(new_data)