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
    from DSPipeline.data_augmentation import ADASYNStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    adasyn_step = ADASYNStep()
    new_data = adasyn_step.fit(data, y_label='y')


