Tree Selection Step
==============================

Uses a tree to select features. Uses sklearn's ExtraTreesRegressor_ (default) and SelectFromModel_ classes.

.. _ExtraTreesRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
.. _SelectFromModel: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html


.. code-block:: python

    DSPipeline.feature_selection.TreeSelectionStep(self, tree_model=ExtraTreesRegressor, tree_kwargs={'n_estimators':100}, select_kwargs={}):

Parameters
----------

+----------------+----------+------------------------------------------+
| **Parameter**  | **Type** | **Description**                          |
+================+==========+==========================================+
| tree_model     | object   | Type of sklearn tree model to use        |
+----------------+----------+------------------------------------------+
| tree_kwargs    | *dict*   | Arguments to pass to sklearn tree model  |
+----------------+----------+------------------------------------------+
| select_kwargs  | *dict*   | Arguments to pass to SelectFromModel     |
+----------------+----------+------------------------------------------+


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
    from DSPipeline.feature_selection import TreeSelectionStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    tree_step = TreeSelectionStep()
    new_X, new_y = tree_step.fit(X, y)
