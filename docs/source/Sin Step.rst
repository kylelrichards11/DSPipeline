Sin Step
========

The Sin Step applies the sine function to specified columns of the given data. It uses numpy's sine function_.

.. _function: https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html


.. code-block:: python

    DSPipeline.data_transformations.SinStep(self, append_input=False, columns=[], kwargs={})

Parameters
----------

+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                                                    |
+===============+==========+====================================================================================================================+
| append_input  | *bool*   | Whether to append the sine features to the given data, or to only keep the transformed data                        |
+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| columns       | *list*   | Columns to apply the sine function to. If the list is empty then sine is applied to all columns (except y_label).  |
+---------------+----------+--------------------------------------------------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to be passed to numpy's sin function                                                                     |
+---------------+----------+--------------------------------------------------------------------------------------------------------------------+


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
    from DSPipeline.data_transformations import SinStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    sin_step = SinStep(columns=['x2'])
    new_X, new_y = sin_step.fit(X, y)
