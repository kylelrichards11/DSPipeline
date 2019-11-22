List Selection Step
===================

Selects columns based on a given list of columns to keep.


.. code-block:: python

    DSPipeline.feature_selection.ListSelectionStep(self, features):

Parameters
----------

+---------------+----------+----------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                |
+===============+==========+================================================================+
| features      | *list*   | List of columns to keep. The *y_label* column is always kept.  |
+---------------+----------+----------------------------------------------------------------+


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
    from DSPipeline.feature_selection import ListSelectionStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    list_step = ListSelectionStep(columns=['x1', 'x2'])
    new_X, new_y = list_step.fit(X, y)
