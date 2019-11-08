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
    from DSPipeline.data_augmentation import SMOTEStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    smote_step = SMOTEStep()
    new_data = smote_step.fit(data, y_label='y')


