Pipeline
========

The Pipeline class stores all of the steps that can be applied to data. It can also be used as a single step containing other sub steps.


.. code-block:: python

    DSPipeline.ds_pipeline.Pipeline(self, steps, append_input=False)

Parameters
----------

+---------------+----------+------------------------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                                |
+===============+==========+================================================================================================+
| steps         | *list*   | Steps that are part of the pipeline                                                            |
+---------------+----------+------------------------------------------------------------------------------------------------+
| append_input  | *bool*   | Whether to append the transformed data to the given data, or to only keep the transformed data |
+---------------+----------+------------------------------------------------------------------------------------------------+

Methods
-------

fit()
``````

.. code-block:: python

    .fit(self, X, y=None, verbose=False)

+---------------+----------------+-----------------------------------------------------------+
| **Parameter** | **Type**       | **Description**                                           |
+===============+================+===========================================================+
| X             | *pd.DataFrame* | Training data                                             |
+---------------+----------------+-----------------------------------------------------------+
| y             | *pd.DataFrame* | Target values                                             |
+---------------+----------------+-----------------------------------------------------------+
| verbose       | *bool*         | Whether or not to output progress of fitting the pipeline |
+---------------+----------------+-----------------------------------------------------------+

**Returns**: *None* or *pd.DataFrame*

transform()
````````````

.. code-block:: python

    .transform(self, X, y=None, allow_sample_removal=True, verbose=False)

+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| **Parameter**          | **Type**       | **Description**                                                                                                                                   |
+========================+================+===================================================================================================================================================+
| X                      | *pd.DataFrame* | Training data                                                                                                                                     |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| y                      | *pd.DataFrame* | Target values                                                                                                                                     |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| allow_sample_removal   | *bool*         | Whether or not the pipeline is allowed to remove any samples from the data frame. Generally this is okay in train data and not okay in test data. |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| verbose                | *bool*         | Whether or not to output progress of fitting the pipeline                                                                                         |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+

**Returns**: *pd.DataFrame*

fit_transform()
``````````````````

.. code-block:: python

    .fit_transform(self, X, y=None, allow_sample_removal=True, verbose=False):

+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| **Parameter**          | **Type**       | **Description**                                                                                                                                   |
+========================+================+===================================================================================================================================================+
| X                      | *pd.DataFrame* | Training data                                                                                                                                     |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| y                      | *pd.DataFrame* | Target values                                                                                                                                     |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| allow_sample_removal   | *bool*         | Whether or not the pipeline is allowed to remove any samples from the data frame. Generally this is okay in train data and not okay in test data. |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| verbose                | *bool*         | Whether or not to output progress of fitting the pipeline                                                                                         |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+

**Returns**: *pd.DataFrame*

Example
-------

.. code-block:: python

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
