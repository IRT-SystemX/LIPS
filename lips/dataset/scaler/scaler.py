"""
Base scaler class
"""
class Scaler(object):
    """
    The base scaler which do nothing
    """
    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def transform(self, x, y):
        return x, y

    def fit_transform(self, x, y):
        return x, y

    def inverse_transform(self, y):
        return y

    def save(self, path):
        pass

    def load(self, path):
        pass
