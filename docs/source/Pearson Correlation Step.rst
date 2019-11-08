Pearson Correlation Step
========================

Uses pearson's correlation to select features. Uses pandas's corr_ method.

.. _corr: https://pandas.pydata.org/pandas-docs/version/0.24/reference/api/pandas.DataFrame.corr.html


.. code-block:: python

    DSPipeline.feature_selection.PearsonCorrStep(self, threshold, kwargs={}):

Parameters
----------

+---------------+----------+-----------------------------------------------------+
| **Parameter** | **Type** | **Description**                                     |
+===============+==========+=====================================================+
| threshold     | *float*  | Minimum correlation value to keep (between 0 and 1) |
+---------------+----------+-----------------------------------------------------+
| kwargs        | *dict*   | Arguments to pass to .corr() function               |
+---------------+----------+-----------------------------------------------------+


Methods
-------

fit()
``````

.. code-block:: python

    .fit(self, data, y_label='label')

+---------------+----------------+-------------------------------------------------------------------------+
| **Parameter** | **Type**       | **Description**                                                         |
+===============+================+=========================================================================+
| data          | *pd.DataFrame* | Training data with labels                                               |
+---------------+----------------+-------------------------------------------------------------------------+
| y_label       | *str*          | Name of the column in data with the known label or value for the sample |
+---------------+----------------+-------------------------------------------------------------------------+

**Returns**: *pd.DataFrame*

transform()
````````````

.. code-block:: python

    .transform(self, data, y_label='label')

+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Parameter**          | **Type**       | **Description**                                                                                                                                               |
+========================+================+===============================================================================================================================================================+
| data                   | *pd.DataFrame* | Data (with or without labels)                                                                                                                                 |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| y_label                | *str*          | Name of the column in data with the known label or value for the sample. If the data is unlabeled (test data) then input what the label of the train data is. |
+------------------------+----------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+

**Returns**: *pd.DataFrame*


Example
-------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from DSPipeline.feature_selection import PearsonCorrStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    corr_step = PearsonCorrStep(threshold=0.10) # The threshold may or may not keep any columns
                                                # depending on the random data generated
    new_data = corr_step.fit(data, y_label='y')
