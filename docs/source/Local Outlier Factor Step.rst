Local Outlier Factor Step
=========================

Uses the local outlier factor to detect and remove outliers. Uses sklearn's LocalOutlierFactor_ class.

.. _LocalOutlierFactor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html


.. code-block:: python

    DSPipeline.outlier_detection.LOFStep(self, include_y=True, kwargs={'contamination': 'auto'}):

Parameters
----------

+---------------+----------+--------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                |
+===============+==========+================================================================================+
| include_y     | *bool*   | Whether or not to include the y data when fitting the LocalOutlierFactor class |
+---------------+----------+--------------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to pass to sklearn's LocalOutlierFactor class                        |
+---------------+----------+--------------------------------------------------------------------------------+


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
    from DSPipeline.outlier_detection import LOFStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    lof_step = LOFStep()
    new_X, new_y = lof_step.fit(X, y)
