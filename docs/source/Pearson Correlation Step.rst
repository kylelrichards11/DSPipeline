Pearson Correlation Step
========================

Uses pearson's correlation to select features. Uses pandas's corr_ method.

.. _corr: https://pandas.pydata.org/pandas-docs/version/0.24/reference/api/pandas.DataFrame.corr.html


.. code-block:: python

    DSPipeline.feature_selection.PearsonCorrStep(self, num_features, kwargs={}):

Parameters
----------

+---------------+----------+----------------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                        |
+===============+==========+========================================================================================+
| num_features  | *float*  | Number of features to keep. If less than 1, then that is the minimum correlation value |
+---------------+----------+----------------------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to pass to .corr() function                                                  |
+---------------+----------+----------------------------------------------------------------------------------------+


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
    from DSPipeline.feature_selection import PearsonCorrStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    corr_step = PearsonCorrStep(num_features=2)
    new_X, new_y = corr_step.fit(X, y)

