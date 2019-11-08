DSPipeline Documentation
======================================

This pacakge is inspired by sklearn's **Pipeline** class_. It extends the capabilities to non-sklearn data manipulation methods. The package consists of **Step** classes which are wrappers for the data transformation technique. Each **Step** object is created with the arguments needed to apply the data transformation method. These **Step** classes represent "remembering" the data transformation so that it can be applied to any given data set.

.. _class: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

Installation
============

.. code-block:: python

    pip install DSPipeline.whl

Guide
^^^^^

.. toctree::
    :maxdepth: 1

    Data Managing
    Data Transformations
    DS Pipeline
    Feature Selection
    Outlier Detection

