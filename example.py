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