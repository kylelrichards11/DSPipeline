class TransformError(Exception):
    def __init__(self):
        """ An error to raise if an attempt is made at transforming data before fitting it """
        self.message = "Must fit before transforming"