Standard Scaler Step
====================

The Standard Scaler Step scales the given data with sklearn's StandardScaler_.

.. _StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html


.. code-block:: python

    DSPipeline.data_transformations.StandardScalerStep(self, append_input=False, kwargs={})

Parameters
----------

+---------------+----------+-----------------------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                               |
+===============+==========+===============================================================================================+
| append_input  | *bool*   | Whether to append the scaled features to the given data, or to only keep the transformed data |
+---------------+----------+-----------------------------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to be passed to sklearn's **StandardScaler** class                                  |
+---------------+----------+-----------------------------------------------------------------------------------------------+


Methods
-------

fit()
``````

.. code-block:: python

    .fit(self, X, y=None)

+---------------+----------------+-----------------+
| **Parameter** | **Type**       | **Description** |
+===============+================+=================+
| X             | *pd.DataFrame* | Training data   |
+---------------+----------------+-----------------+
| y             | *pd.DataFrame* | Target values   |
+---------------+----------------+-----------------+

**Returns**: *pd.DataFrame*

transform()
````````````

.. code-block:: python

    .transform(self, X, y=None)

+----------------+----------------+-----------------+
| **Parameter**  | **Type**       | **Description** |
+================+================+=================+
| X              | *pd.DataFrame* | Training data   |
+----------------+----------------+-----------------+
| y              | *pd.DataFrame* | Target values   |
+----------------+----------------+-----------------+

**Returns**: *pd.DataFrame*


Example
-------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from DSPipeline.data_transformations import StandardScalerStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    scale_step = StandardScalerStep()
    new_X, new_y = scale_step.fit(X, y)
