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
    from DSPipeline.outlier_detection import ABODStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    abod_step = ABODStep(1)
    new_X, new_y = abod_step.fit(X, y)
