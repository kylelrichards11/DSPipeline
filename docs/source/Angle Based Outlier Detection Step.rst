Angle Based Outlier Detection Step
==================================

Uses angle based outlier detection to detect and remove outliers. Uses pyod's ABOD_ class.

.. _ABOD: https://pyod.readthedocs.io/en/latest/_modules/pyod/models/abod.html


.. code-block:: python

    DSPipeline.outlier_detection.ABODStep(self, num_remove, kwargs={}):

Parameters
----------

+----------------+----------+----------------------------------------+
| **Parameter**  | **Type** | **Description**                        |
+================+==========+========================================+
| num_remove     | *int*    | Number of detected outliers to remove  |
+----------------+----------+----------------------------------------+
| select_kwargs  | *dict*   | Arguments to pass to pyod's ABOD class |
+----------------+----------+----------------------------------------+


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
    from DSPipeline.outlier_detection import ABODStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    abod_step = ABODStep(1)
    new_data = abod_step.fit(data, y_label='y')

