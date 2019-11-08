Isolation Forest Step
=====================

Uses an isolation forest to detect and remove outliers. Uses sklearn's IsolationForest_ class.

.. _IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html


.. code-block:: python

    DSPipeline.outlier_detection.IsoForestStep(self, include_y=True, kwargs={'contamination': 'auto', 'behaviour': 'new'}):

Parameters
----------

+---------------+----------+------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                        |
+===============+==========+========================================================================+
| include_y     | *bool*   | Whether or not to include the y data when fitting the isolation forest |
+---------------+----------+------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to pass to sklearn's IsolationForest class                   |
+---------------+----------+------------------------------------------------------------------------+


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
    from DSPipeline.outlier_detection import IsoForestStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    iso_step = IsoForestStep()
    new_data = iso_step.fit(data, y_label='y')