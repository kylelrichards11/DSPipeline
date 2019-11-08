Pipeline
========

The Pipeline class stores all of the steps that can be applied to data.


.. code-block:: python

    DSPipeline.ds_pipeline.Pipeline(self, steps)

Parameters
----------

+---------------+----------+-------------------------------------+
| **Parameter** | **Type** | **Description**                     |
+===============+==========+=====================================+
| steps         | *list*   | Steps that are part of the pipeline |
+---------------+----------+-------------------------------------+

Methods
-------

fit()
``````

.. code-block:: python

    .fit(self, data, y_label='label', verbose=False)

+---------------+----------------+-------------------------------------------------------------------------+
| **Parameter** | **Type**       | **Description**                                                         |
+===============+================+=========================================================================+
| data          | *pd.DataFrame* | Training data with labels                                               |
+---------------+----------------+-------------------------------------------------------------------------+
| y_label       | *str*          | Name of the column in data with the known label or value for the sample |
+---------------+----------------+-------------------------------------------------------------------------+
| verbose       | *bool*         | Whether or not to output progress of fitting the pipeline               |
+---------------+----------------+-------------------------------------------------------------------------+

**Returns**: *None* or *pd.DataFrame*

transform()
````````````

.. code-block:: python

    .transform(self, data, y_label='label', allow_sample_removal=True, verbose=False)

+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Parameter**          | **Type**       | **Description**                                                                                                                                               |
+========================+================+===============================================================================================================================================================+
| data                   | *pd.DataFrame* | Data (with or without labels)                                                                                                                                 |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| y_label                | *str*          | Name of the column in data with the known label or value for the sample. If the data is unlabeled (test data) then input what the label of the train data is. |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| allow_sample_removal   | *bool*         | Whether or not the pipeline is allowed to remove any samples from the data frame. Generally this is okay in train data and not okay in test data.             |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| verbose                | *bool*         | Whether or not to output progress of fitting the pipeline                                                                                                     |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+

**Returns**: *pd.DataFrame*

fit_transform()
``````````````````

.. code-block:: python

    fit_transform(self, data, y_label='label', allow_sample_removal=True, verbose=False):

+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| **Parameter**          | **Type**       | **Description**                                                                                                                                   |
+========================+================+===================================================================================================================================================+
| data                   | *pd.DataFrame* | Training data with labels                                                                                                                         |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| y_label                | *str*          | Name of the column in data with the known label or value for the sample                                                                           |
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
    from DSPipeline.data_managing import split_x_y
    from DSPipeline.outlier_detection import ABODStep

    # Other Imports
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

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