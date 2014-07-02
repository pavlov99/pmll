import unittest
import math

from .base import QualityMeasurerLinear, QualityMeasurerNominal


class TestQualityMeasurer(unittest.TestCase):

    """ Test quality metrics calculation."""

    def test_measure_linear(self):
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

    def test_measure_nominal(self):
        result = QualityMeasurerNominal()
        predicted = (x for x in [1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        actual = (x for x in [1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
        for p, a in zip(predicted, actual):
            result.append(p, a)

        self.assertEqual(result.tp, 2)
        self.assertEqual(result.tn, 4)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.fn, 3)

        self.assertEqual(result.tpr, 0.4)
        self.assertEqual(result.tpr, result.sensitivity)
        self.assertEqual(result.tnr, 0.8)
        self.assertEqual(result.tnr, result.specificity)

        self.assertEqual(result.precision, 2.0 / 3)
        self.assertEqual(result.recall, result.sensitivity)

        self.assertEqual(result.accuracy, 0.6)
        self.assertEqual(result.f1, 0.5)
        self.assertAlmostEqual(result.f(2), 4 / 9.2, 15)
        self.assertEqual(result.mcc, 1.0 / math.sqrt(21))
