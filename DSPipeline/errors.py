class Transform_Error(Exception):
    def __init__(self):
        self.message = "Must fit before transforming"