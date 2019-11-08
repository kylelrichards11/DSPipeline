Lasso Selection Step
====================

Uses lasso regularization to select features. Uses sklearn's **Lasso_** and **SelectKBest_** classes.

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
    from DSPipeline.feature_selection import LassoSelectionStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    lasso_step = LassoSelectionStep(select_kwargs={'k':2})
    new_data = lasso_step.fit(data, y_label='y')
