Lasso Selection Step
====================

Uses lasso regularization to select features. Uses sklearn's Lasso_ and SelectKBest_ classes.

.. _Lasso: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
.. _SelectKBest: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html


.. code-block:: python

    DSPipeline.feature_selection.LassoSelectionStep(self, num_remove, kwargs={}):

Parameters
----------

+----------------+----------+-------------------------------------------------------+
| **Parameter**  | **Type** | **Description**                                       |
+================+==========+=======================================================+
| lasso_kwargs   | *dict*   | Arguments to be passed to sklearn's Lasso class       |
+----------------+----------+-------------------------------------------------------+
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
    from DSPipeline.feature_selection import LassoSelectionStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    lasso_step = LassoSelectionStep(select_kwargs={'k':2})
    new_X, new_y = lasso_step.fit(X, y)
