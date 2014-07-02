import unittest
import math

from ..data import Data
from .base import measure  # , measure_linear, measure_nominal


class TestMetrics(unittest.TestCase):

    """ Test quality metrics calculation."""

    def test_measure(self):
        data_predicted = Data([[0], [5]])
        data_actual = Data([[2], [4]])
        result = measure(
            data_predicted, data_actual, feature=data_predicted.features[0])
        #result = measure_linear(
            #(x for x in [0, 5]),
            #(x for x in [2, 4]),
        #)

        self.assertEqual(result.mse, 2.5)
        self.assertEqual(result.mae, 1.5)
        self.assertEqual(result.rmse, math.sqrt(2.5))
        self.assertEqual(result.nrmse, math.sqrt(2.5) / 2)
        self.assertEqual(result.cvrmse, math.sqrt(2.5) / 3)
