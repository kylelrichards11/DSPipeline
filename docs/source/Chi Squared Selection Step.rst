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
    from DSPipeline.feature_selection import ChiSqSelectionStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    chi2_step = ChiSqSelectionStep(select_kwargs={'k':2})
    new_data = chi2_step.fit(data, y_label='y')
