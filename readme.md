# Data Science Pipeline
## Overview
This pacakge is inspired by `sklearn`'s `pipeline` [class](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). It extends the capabilities to non-sklearn data manipulation methods. The package consists of `_Step` classes which are wrappers for the data transformation technique. Each `_Step` object is created with the arguments needed to apply the data transformation method. These `_Step` classes represent "remembering" the data transformation so that it can be applied to any given data set.

## Adding a Pipeline Step
To add a step, first create a class for it in the appropriate file. 

The constructor must contain all information that needs to be remembered to recreate the step. All step specific arguments must pass through here. Additionally, a field stating that the model has been fitted must be created. Finally `self.test_data` specifies whether or not to run this step on test data (`True` for all steps besides outlier detection).

`fit_transform` and `transform` methods must be created, each of which must have the signature `func(self, data, y_label='label')`. `fit_transform` should take care of anything that is done based on the train data, while `transform` needs to be able to run on any input data set. Note that when `transform` is called on a test set, there are no labels. There is no `fit` function because when fitting in the pipeline, you must fit each step to the output from the previous steps. Therefore in order to fit a step, the data must have been transformed by the previous steps. 

## Applying the Data Transformations
`_Step` objects are created and then put into a list in the desired order of the transformations. This list of steps is passed into the `DS_Pipeline` class to create the pipeline object. For the first use, the pipeline must be fit to a dataset (generally the training data). Then any subsequent datasets can be fit with the same features as the first dataset.

## Example Use
```python
# DS_Pipeline Imports
from DSPipeline.data_managing import split_X_y
from DSPipeline.data_transformations import Standard_Scaler_Step
from DSPipeline.outlier_detection import ABOD_Step
from DSPipeline.feature_selection import Pearson_Corr_Step
from DSPipeline.ds_pipeline import Pipeline

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

X_data = pd.DataFrame(boston.data, columns=boston.feature_names)
y_data = pd.DataFrame(boston.target, columns=[y_label])
data = pd.concat((X_data, y_data), axis=1)

# Split into test and train. 
# NOTE: Resetting the indices is very important and not doing so will result in errors
train, test = train_test_split(data)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
test_X, test_y = split_X_y(test, y_label=y_label)

# Create Steps
scale_step = Standard_Scaler_Step()
abod_step = ABOD_Step(num_remove=5, kwargs={'contamination':0.05})
corr_step = Pearson_Corr_Step(threshold=0.25)

# Make Pipeline
pipeline_steps = [scale_step, corr_step]
pipeline = Pipeline(pipeline_steps)

# Transform data sets
train_transformed = pipeline.fit_transform(train, y_label=y_label)
test_X_transformed = pipeline.transform(test_X, allow_sample_removal=False)

# Use data to make predictions
train_X, train_y = split_X_y(train_transformed, y_label=y_label)
model = LinearRegression()
model.fit(train_X, train_y)
y_hat = model.predict(test_X_transformed)
print(f'MAE: {mean_absolute_error(test_y, y_hat):.3f}')
```
