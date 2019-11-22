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
    from DSPipeline.outlier_detection import IsoForestStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')
    iso_step = IsoForestStep()
    new_X, new_y = iso_step.fit(X, y)
