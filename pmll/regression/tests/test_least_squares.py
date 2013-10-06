import numpy as np
import unittest

from ...base import BaseRegressor
from ...data import Data
from ...feature import FeatureLin
from ..least_squares import LeastSquaresModel


class LeastSquaresModelTest(unittest.TestCase):
    def setUp(self):
        self.model = LeastSquaresModel
        self.data = Data([
            [0, 0, 1],
            [1, 1, 2],
            [2, 2, 0],
        ], features=[FeatureLin("f{0}".format(i)) for i in range(3)])
        self.labels = self.data[:, 0]
        self.objects = self.data[:, 1:]

    def test_fitted_model(self):
        weights = np.matrix([[1],
                             [1]])
        model = self.model(weights=weights)
        regressor = model.predictor
        self.assertTrue(isinstance(regressor, BaseRegressor))
        result = regressor.regress(self.objects)
        self.assertTrue(isinstance(result, Data))
        self.assertEqual(result, Data([[1], [3], [2]], result.features))

    def test_train(self):
        regressor = self.model().train(self.objects, self.labels).predictor
        self.assertTrue(isinstance(regressor, BaseRegressor))
        self.assertTrue((regressor.weights == np.matrix([[1], [0]])).all())
        result = regressor.regress(self.objects)
        self.assertTrue(isinstance(result, Data))
        self.assertEqual(result, Data([[0], [1], [2]], result.features))

#if __name__ == '__main__':
    #ls = LeastSquaresModel
    #data = np.genfromtxt(open("bread.csv", "rb"), dtype=float, delimiter=',')
    #y = np.asmatrix(data[:, 0][:, np.newaxis])
    #x = np.asmatrix(data[:, 1][:, np.newaxis])

    ## import matplotlib.pyplot as plt
    ## plt.plot(x, y, "k.")
    ## plt.show()

    #model = LeastSquaresPolynomModel()
    #for is_add_constant in [0, 1]:
        #for max_power in range(1, 10):
            #print "Add contsant = %s. Max power = %s" % (is_add_constant,
                                                         #max_power)
            #model.train(x, y, is_add_constant=1, max_power=max_power)
            #print model.weights
            #print "Train time:\t", model.train_time

            ## import matplotlib.pyplot as plt
            ## plt.plot(y, "k.")
            ## plt.plot(x * self.weights, "b")
            ## plt.show()
            ## print "Error:\t\t", float(sum(map(lambda x: x ** 2,
            ## y - x * self.weights)))
