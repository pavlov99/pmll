import numpy as np
import unittest

class IrlsModel(object):
    """
    Iterative reweighted least squares regression model
    Model is used to choose parameters using given data.
    """
    def __init__(self, weights=None):
        # 'weights' are used for 1) init weights 2) create model
        self.weights = weights
        self.__history = {'weights': self.weights}

    @staticmethod
    def __logit(z):
        return 1 / (1 + np.exp(-z))

    def train(self):
        pass


class IrlsClassifier(object):
    """
    Iterative reweighted least squares classifier.
    """
    def __init__(self, model):
        self.model = model


class TestIrlsModel(unittest.TestCase):
    pass


class TestIrlsClassifier(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
