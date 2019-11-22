Data Managing
=============

split_x_y()
-----------

Splits the input data frame into X and y components.

.. code-block:: python

    split_x_y(X, y=None)


**Parameters**

+---------------+--------------------+---------------------------------------+
| **Parameter** | **Type**           | **Description**                       |
+===============+====================+=======================================+
| data          | *pandas.DataFrame* | Data to split                         |
+---------------+--------------------+---------------------------------------+
| y_label       | *string*           | Name of the y column in the DataFrame |
+---------------+--------------------+---------------------------------------+

**Returns**: (*pandas.DataFrame*, *pandas.DataFrame*)

**Example**

.. code-block:: python

    import numpy as np
    import pandas as pd
    from DSPipeline.data_managing import split_x_y

    data = pd.DataFrame(np.random.uniform(size=(10, 4)), columns=['x1', 'x2', 'x3', 'y'])
    X_data, y_data = split_x_y(data, y_label='y')

