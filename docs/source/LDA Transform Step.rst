LDA Step
========

The LDA Step applies linear discriminant analysis to the given data with sklearn's LinearDiscriminantAnalysis_.

.. _LinearDiscriminantAnalysis: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html


.. code-block:: python

    DSPipeline.data_transformations.LDATransformStep(self, append_input=False, kwargs={})

Parameters
----------

+---------------+----------+-------------------------------------------------------------------------------------------------------+
| **Parameter** | **Type** | **Description**                                                                                       |
+===============+==========+=======================================================================================================+
| append_input  | *bool*   | Whether to append the calculated LDA features to the given data, or to only keep the transformed data |
+---------------+----------+-------------------------------------------------------------------------------------------------------+
| kwargs        | *dict*   | Arguments to be passed to sklearn's **LinearDiscriminantAnalysis** class                              |
+---------------+----------+-------------------------------------------------------------------------------------------------------+


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
    from DSPipeline.data_transformations import LDATransformStep

    X = pd.DataFrame(np.random.uniform(size=(20, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.randint(3, size=20), name='y')

    lda_step = LDATransformStep(append_input=True, kwargs={'n_components':2})
    new_X, new_y = lda_step.fit(X, y)
