ADASYN Step
===========

Adaptive Synthetic Over-Sampling Technique (ADASYN) to create balanced samples. Uses imblearn's ADASYN_ class.

.. _ADASYN: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html


.. code-block:: python

    DSPipeline.data_augmentation.ADASYNStep(self, kwargs={}):

Parameters
----------

+---------------+----------+--------------------------------------------+
| **Parameter** | **Type** | **Description**                            |
+===============+==========+============================================+
| kwargs        | *dict*   | Arguments to pass to imblearn adasyn class |
+---------------+----------+--------------------------------------------+


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
    from DSPipeline.data_augmentation import ADASYNStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    adasyn_step = ADASYNStep()
    new_X, new_y = adasyn_step.fit(X, y)
