""" Measure quality of prediction.

Methods used:

Classification:
    TPR - True positive rate
    FPR - False positive rate
    TNR - True negative rate
    FNR - False negative rate

Regression:
    MSE - mean squared error (SSE sum of squared errors)
    MAE - mean absolute error (SAE sum of absolute errors)
    RMSE - root-mean-square error
    NRMSE - normalized root-mean-square error: RMSE / (x_max - x_min)
    CVRMSE - Coefficient of variation of the RMSE: RMSE / mean(x)

"""
import math


class QualityMeasurerLinear(object):
    def __init__(self):
        self.sse = 0.0
        self.sae = 0.0
        self._number_objects = 0
        self._actual_sum = 0.0
        self._actual_min = None
        self._actual_max = None

    @property
    def _actual_mean(self):
        return self._actual_sum / self._number_objects

    @property
    def mse(self):
        return self.sse / self._number_objects

    @property
    def mae(self):
        return self.sae / self._number_objects

    @property
    def rmse(self):
        return math.sqrt(self.mse)

    @property
    def nrmse(self):
        return self.rmse / (self._actual_max - self._actual_min)

    @property
    def cvrmse(self):
        return self.rmse / self._actual_mean

    def append(self, predicted, actual):
        self._number_objects += 1
        self._actual_sum += actual

        if self._actual_max is None or actual > self._actual_max:
            self._actual_max = actual

        if self._actual_min is None or actual < self._actual_min:
            self._actual_min = actual

        self.sse += (predicted - actual) ** 2
        self.sae += abs(predicted - actual)


def measure(data_predicted, data_actual, feature=None):
    predicted = data_predicted[:, data_predicted.features.index(feature)]
    actual = data_actual[:, data_actual.features.index(feature)]
    quality = QualityMeasurerLinear()

    for p, a in zip(predicted, actual):
        quality.append(p[0], a[0])
    return quality
