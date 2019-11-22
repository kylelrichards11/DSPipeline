Poly Step
=========

The Poly Step applies polynomial feature combinations to the given data with sklearn's PolynomialFeatures_.

.. _PolynomialFeatures: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html


.. code-block:: python

    DSPipeline.data_transformations.PolyStep(self, append_input=False, kwargs={})

Parameters
----------

+---------------+----------+---------------------------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                                   |
+===============+==========+===================================================================================================+
| append_input  | *bool*   | Whether to append the polynomial features to the given data, or to only keep the transformed data |
+---------------+----------+---------------------------------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to be passed to sklearn's **PolynomialFeatures** class                                  |
+---------------+----------+---------------------------------------------------------------------------------------------------+


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
    from DSPipeline.data_transformations import PolyStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    poly_step = PolyStep(kwargs={'degree':3})
    new_X, new_y = poly_step.fit(X, y)
