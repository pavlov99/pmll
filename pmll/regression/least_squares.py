import numpy as np

from ..base import BaseModel, BaseRegressor
from ..data import Data
from ..feature import FeatureLin


class FeatureGenerator(object):
    def __init__(self, x):
        self.x = x

    def add_polynomial(self, max_power=1):
        self.x = np.hstack(np.power(self.x, power) for power
                           in range(1, max_power + 1))

    def add_constant(self):
        self.x = np.hstack([np.ones([self.x.shape[0], 1]), self.x])


class LeastSquaresModel(BaseModel):
    def __init__(self, weights=None):
        super(self.__class__, self).__init__()
        self.weights = weights

    def train(self, x, y):
        x, y = x.matrix, y.matrix
        self.weights = (x.T * x).I * x.T * y
        return self

    @property
    def predictor(self):
        return LeastSquaresRegressor(self.weights)


class LeastSquaresPolynomModel(BaseModel):
    def __init__(self, weights=None):
        super(self.__class__, self).__init__()
        self.weights = weights

    def train(self, x, y, max_power=1, is_add_constant=0):
        # Add polynomial features
        x = np.hstack(np.power(x, power) for power in range(1, max_power + 1))

        if is_add_constant:
            x = np.hstack([np.ones([x.shape[0], 1]), x])

        self.weights = (x.T * x).I * x.T * y


class LeastSquaresRegressor(BaseRegressor):
    def __init__(self, weights, feature_transformator=None):
        self.weights = weights
        self.feature_transformator = feature_transformator or (lambda x: x)

    def regress(self, x):
        result = self.feature_transformator(x).matrix * self.weights
        return Data(
            result.tolist(),
            [FeatureLin("f" + str(i)) for i in range(result.shape[1])])
