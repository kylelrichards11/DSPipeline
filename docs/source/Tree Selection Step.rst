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
    from DSPipeline.feature_selection import TreeSelectionStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    reg_step = TreeSelectionStep()
    new_data = reg_step.fit(data, y_label='y')