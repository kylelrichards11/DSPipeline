PCA Step
========

The PCA Step applies principal component analysis to the given data with sklearn's PCA_.

.. _PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html


.. code-block:: python

    DSPipeline.data_transformations.PCAStep(self, append_input=False, kwargs={})

Parameters
----------

+---------------+----------+---------------------------------------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                                               |
+===============+==========+===============================================================================================================+
| append_input  | *bool*   | Whether to append the calculated principal components to the given data, or to only keep the transformed data |
+---------------+----------+---------------------------------------------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to be passed to sklearn's **PCA** class                                                             |
+---------------+----------+---------------------------------------------------------------------------------------------------------------+


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
    from DSPipeline.data_transformations import PCAStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    pca_step = PCAStep(append_input=True, kwargs={'n_components':2})
    new_X, new_y = pca_step.fit(X, y)
