SMOTE Step
==========

Uses Synthetic Minority Over-Sampling Technique (SMOTE) to create balanced samples. Uses imblearn's SMOTE_ family of classes.

.. _SMOTE: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html


.. code-block:: python

    DSPipeline.data_augmentation.SMOTEStep(self, smote_class=SMOTE, kwargs={}):

Parameters
----------

+---------------+----------------------+---------------------------------------------+
| **Parameter** | **Type**             | **Description**                             |
+===============+======================+=============================================+
| smote_class   | imblearn smote class | Which of imblearn's smote variations to use |
+---------------+----------------------+---------------------------------------------+
| kwargs        | *dict*               | Arguments to pass to imblearn smote class   |
+---------------+----------------------+---------------------------------------------+


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
    from DSPipeline.data_augmentation import SMOTEStep

    X = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(np.random.uniform(size=10), name='y')

    smote_step = SMOTEStep()
    new_X, new_y = smote_step.fit(X, y)


