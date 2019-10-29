# AML Task 1
## Overview
My code is inspired by `sklearn`'s `pipeline` [class](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Basically you make a pipeline as instructions to modify the data before using a model. There are currently three types of steps, `data_transformation`, `feature_selection`, and `outlier_detection`. Once a pipeline is created, then it must be fit on the train data (`pipeline.fit()`). Calling `transform()` is a little different because there are two types of transform, one for train data and one for test/validation data. This comes from the fact that we want to remove outliers from the train data, but cannot do that for test data. 

## Adding a Pipeline Step
To add a step, first create a class for it in the appropriate file. 

The constructor must contain all information that needs to be remembered to recreate the step. All step specific arguments must pass through here. Additionally, a field stating that the model has been fitted must be created. Finally `self.test_data` specifies whether or not to run this step on test data (`True` for all steps besides outlier detection).

`fit_transform` and `transform` methods must be created, each of which must have the signature `func(self, data, y_label='label')`. `fit_transform` should take care of anything that is done based on the train data, while `transform` needs to be able to run on any input data set. Note that when `transform` is called on a test set, there are no labels. There is no `fit` function because when fitting in the pipeline, you must fit each step to the output from the previous steps. Therefore in order to fit a step, the data must have been transformed by the previous steps. 