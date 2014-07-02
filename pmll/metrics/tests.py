import unittest
import math

from .base import QualityMeasurerLinear


class TestQualityMeasurerLinear(unittest.TestCase):

    """ Test quality metrics calculation."""

    def test_measure(self):
        result = QualityMeasurerLinear()
        predicted = (x for x in [0, 5])
        actual = (x for x in [2, 4])
        for p, a in zip(predicted, actual):
            result.append(p, a)

        self.assertEqual(result.mse, 2.5)
        self.assertEqual(result.mae, 1.5)
        self.assertEqual(result.rmse, math.sqrt(2.5))
        self.assertEqual(result.nrmse, math.sqrt(2.5) / 2)
        self.assertEqual(result.cvrmse, math.sqrt(2.5) / 3)
