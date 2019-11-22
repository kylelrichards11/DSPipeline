Chi Squared Selection Step
==========================

Uses the chi squared test to select relevant features for classification tasks. Uses sklearn's chi2_ and SelectKBest_ classes.

.. _chi2: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
.. _SelectKBest: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html


.. code-block:: python

    DSPipeline.feature_selection.ChiSqSelectionStep(self, select_kwargs={}):

Parameters
----------

+----------------+----------+-------------------------------------------------------+
| **Parameter**  | **Type** | **Description**                                       |
+================+==========+=======================================================+
| select_kwargs  | *dict*   | Arguments to be passed to sklearn's SelectKBest class |
+----------------+----------+-------------------------------------------------------+


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
    from DSPipeline.feature_selection import ChiSqSelectionStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.randint(2, size=10), name='y')

    chi2_step = ChiSqSelectionStep(select_kwargs={'k':2})
    new_X, new_y = chi2_step.fit(X, y)
