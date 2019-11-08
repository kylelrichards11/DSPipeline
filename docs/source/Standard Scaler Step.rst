Standard Scaler Step
====================

The Standard Scaler Step scales the given data with sklearn's StandardScaler_.

.. _StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html


.. code-block:: python

    DSPipeline.data_transformations.StandardScalerStep(self, kwargs={})

Parameters
----------

+---------------+----------+--------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                              |
+===============+==========+==============================================================+
| kwargs        | *dict*   | Arguments to be passed to sklearn's **StandardScaler** class |
+---------------+----------+--------------------------------------------------------------+


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
    from DSPipeline.data_transformations import StandardScalerStep

    data = pd.DataFrame(np.random.uniform(size=(10, 3)), columns=['x1', 'x2', 'y'])
    scaled_data = StandardScalerStep().fit(data, y_label='y')
