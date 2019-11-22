[![Build Status](https://travis-ci.com/kylelrichards11/DSPipeline.svg?token=wqDVpwhQq3xYjNDgN9tk&branch=master)](https://travis-ci.com/kylelrichards11/DSPipeline) [![codecov](https://codecov.io/gh/kylelrichards11/DSPipeline/branch/master/graph/badge.svg?token=5QP9hGZm8O)](https://codecov.io/gh/kylelrichards11/DSPipeline) [![Documentation Status](https://readthedocs.org/projects/dspipeline/badge/?version=latest)](https://dspipeline.readthedocs.io/en/latest/?badge=latest)

# Data Science Pipeline
## Overview
This pacakge is inspired by `sklearn`'s `pipeline` [class](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). It extends the capabilities to non-sklearn data manipulation methods. The package consists of `Step` classes which are wrappers for the data transformation technique. Each `Step` object is created with the arguments needed to apply the data transformation method. These `Step` classes represent "remembering" the data transformation so that it can be applied to any given data set.

## Documentation
Documentation can be found at [https://dspipeline.readthedocs.io/en/latest/](https://dspipeline.readthedocs.io/en/latest/).

## Applying the Data Transformations
`Step` objects are created and then put into a list in the desired order of the transformations. This list of steps is passed into the `Pipeline` class to create the pipeline object. For the first use, the pipeline must be fit to a dataset (generally the training data). Then any subsequent datasets can be fit with the same features as the first dataset.

## Example Use
```python
# DSPipeline Imports
from DSPipeline.ds_pipeline import Pipeline
from DSPipeline.data_transformations import StandardScalerStep
from DSPipeline.feature_selection import PearsonCorrStep
from DSPipeline.outlier_detection import ABODStep

# Other Imports
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

# Load Data
boston = load_boston()
y_label = "MEDV"    # The traditional name for Boston's target value

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name=y_label)

# Split into test and train. 
# NOTE: Resetting the indices is very important and not doing so will result in errors
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33)
train_X = train_X.reset_index(drop=True)
test_X = test_X.reset_index(drop=True)
train_y = train_y.reset_index(drop=True)
test_y = test_y.reset_index(drop=True)

# Create Steps
scale_step = StandardScalerStep()
abod_step = ABODStep(num_remove=5, kwargs={'contamination':0.05})
corr_step = PearsonCorrStep(num_features=0.25)

# Make Pipeline
pipeline_steps = [scale_step, abod_step, corr_step]
pipeline = Pipeline(pipeline_steps)

# Transform data sets
train_X_transformed, train_y_transformed = pipeline.fit_transform(train_X, train_y)
test_X_transformed = pipeline.transform(test_X, allow_sample_removal=False)

# Use data to make predictions
model = LinearRegression()
model.fit(train_X_transformed, train_y_transformed)
y_hat = model.predict(test_X_transformed)
print(f'MAE: {mean_absolute_error(test_y, y_hat):.3f}')
```

## Adding a Pipeline Step
To add a step, first create a class for it in the appropriate file. 

The constructor must contain all information that needs to be remembered to recreate the step. All step specific arguments must pass through here. Additionally, a field stating that the model has been fitted must be created. This is used in the `transform` function to throw a `TransformError` if the step has not been previously fitted. Finally `self.changes_num_samples` specifies whether or not running this step will add or remove data samples. Generally this is okay for training data but not for testing data.

`fit` and `transform` methods must be created, each of which must have the signature `func(self, X, y=None)`. `fit` should take care of anything that is done based on the train data, while `transform` needs to be able to run on any input data set. Note that when `transform` is called on a test set, there are no labels. The `fit` function must also return the transformed data because when fitting in the pipeline, each step must be fitted to the output from the previous steps. Therefore in order to fit a step, the data must have been transformed by the previous steps. 

## Example Adding
```python
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
```

### Output

```
Fitting Select features with 'a'
Transforming Select features with 'a'
      apple    banana      date  eggplant     grape         y
0  3.416895  0.871671  8.273050  8.790200  7.514323  2.314195
1  7.412230  7.281992  3.188810  3.872916  8.981018  7.936679
2  8.967431  1.580992  9.227848  1.920189  3.145534  4.457451
3  6.543678  8.089490  1.338272  0.186144  3.652871  5.901529
4  7.673827  3.143026  9.678693  3.899328  4.019177  2.502145
```

# Testing
To run all tests simply call

```
pytest test.py
```

Tests for individual files are in the `tests/` directory.
