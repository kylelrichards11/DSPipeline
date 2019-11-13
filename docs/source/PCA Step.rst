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
    from DSPipeline.data_transformations import PCAStep

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    pca_step = PCAStep(append_input=True, kwargs={'n_components':2})
    data_with_pca = pca_step.fit(data, y_label='y')
