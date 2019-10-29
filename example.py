# DSPipeline Imports
from DSPipeline.ds_pipeline import Pipeline
from DSPipeline.data_transformations import StandardScalerStep
from DSPipeline.feature_selection import PearsonCorrStep
from DSPipeline.data_managing import split_x_y
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

X_data = pd.DataFrame(boston.data, columns=boston.feature_names)
y_data = pd.DataFrame(boston.target, columns=[y_label])
data = pd.concat((X_data, y_data), axis=1)

# Split into test and train. 
# NOTE: Resetting the indices is very important and not doing so will result in errors
train, test = train_test_split(data)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
test_X, test_y = split_x_y(test, y_label=y_label)

# Create Steps
scale_step = StandardScalerStep()
abod_step = ABODStep(num_remove=5, kwargs={'contamination':0.05})
corr_step = PearsonCorrStep(threshold=0.25)

# Make Pipeline
pipeline_steps = [scale_step, abod_step, corr_step]
pipeline = Pipeline(pipeline_steps)

# Transform data sets
train_transformed = pipeline.fit_transform(train, y_label=y_label)
test_X_transformed = pipeline.transform(test_X, allow_sample_removal=False)

# Use data to make predictions
train_X, train_y = split_x_y(train_transformed, y_label=y_label)
model = LinearRegression()
model.fit(train_X, train_y)
y_hat = model.predict(test_X_transformed)
print(f'MAE: {mean_absolute_error(test_y, y_hat):.3f}')