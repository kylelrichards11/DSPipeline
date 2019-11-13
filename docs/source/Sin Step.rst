Sin Step
========

The Sin Step applies the sine function to specified columns of the given data. It uses numpy's sine function_.

.. _function: https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html


.. code-block:: python

    DSPipeline.data_transformations.SinStep(self, append_input=False, columns=[], kwargs={})

Parameters
----------

+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                                                    |
+===============+==========+====================================================================================================================+
| append_input  | *bool*   | Whether to append the sine features to the given data, or to only keep the transformed data                        |
+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| columns       | *list*   | Columns to apply the sine function to. If the list is empty then sine is applied to all columns (except y_label).  |
+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to be passed to numpy's sin function                                                                     |
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
    from DSPipeline.data_transformations import SinStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    sin_step = SinStep(columns=['x2'])
    new_data = sin_step.fit(data, y_label='y')
