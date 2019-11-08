Log Step
========

The Log Step applies a logarithmic function to specified columns of the given data. The default uses numpy's log function_.

.. _function: https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html


.. code-block:: python

    DSPipeline.data_transformations.LogStep(self, append_data=False, columns=None, log_func=np.log, kwargs={}):

Parameters
----------

+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                                                    |
+===============+==========+====================================================================================================================+
| append_data   | *bool*   | Whether to append the log features to the given data, or to only keep the transformed data                         |
+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| columns       | *list*   | Columns to apply the log function to. If the list is empty then log is applied to all columns (except y_label).    |
+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| log_func      | function | The type of log to apply. Default is natural log (ln).                                                             |
+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to be passed to given log function                                                                       |
+---------------+----------+--------------------------------------------------------------------------------------------------------------------+


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
    from DSPipeline.data_transformations import LogStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    log_step = LogStep(columns=['x2', 'x3'], log_func=np.log10)
    new_data = log_step.fit(data, y_label='y')
