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

    """ Quality measurer for linear features.

    .. versionadded:: 0.1.5

    """

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


class QualityMeasurerNominal(object):

    """ Quality measurer for nominal features.

    .. versionadded:: 0.1.5

    """

    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    @property
    def tpr(self):
        """ True Positive Rate.

        Also known as sensitivity or recall.
        Ratio of correct objects identified as correct.

        """
        return float(self.tp) / (self.tp + self.fn)

    @property
    def tnr(self):
        """ True Negative Rate.

        Also known as specificity.

        """
        return float(self.tn) / (self.fp + self.tn)

    @property
    def specificity(self):
        return self.tnr

    @property
    def sensitivity(self):
        return self.tpr

    @property
    def recall(self):
        return self.tpr

    @property
    def precision(self):
        return float(self.tp) / (self.tp + self.fp)

    @property
    def accuracy(self):
        return float(self.tp + self.tn) /\
            (self.tp + self.tn + self.fp + self.fn)

    def f(self, beta):
        """ F measure.

        Measure of a test's accuracy. It considers both the precision and the
        recall of the test to compute the score.

        F(b) = (1 + b^2) * precision * recall / (b^2 * precision + recall)

        """
        b = beta ** 2
        return float(1 + b) * self.tp / (
            (1 + b) * self.tp + b * self.fn + self.fp)

    @property
    def f1(self):
        return self.f(1)

    @property
    def mcc(self):
        """ Matthews correlation coefficient.

        The Matthews correlation coefficient is used in machine learning as
        a measure of the quality of binary (two-class) classifications. It
        takes into account true and false positives and negatives and is
        generally regarded as a balanced measure which can be used even if the
        classes are of very different sizes. The MCC is in essence a
        correlation coefficient between the observed and predicted binary
        classifications.

        """
        return float(self.tp * self.tn - self.fp * self.fn) / math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) *
            (self.tn + self.fn))

    def append(self, predicted, actual):
        if predicted == actual:
            self.tp += int(bool(predicted))
            self.tn += int(not bool(predicted))
        else:
            self.fp += int(bool(predicted))
            self.fn += int(not bool(predicted))


def measure(data_predicted, data_actual, feature=None):
    predicted = data_predicted[:, data_predicted.features.index(feature)]
    actual = data_actual[:, data_actual.features.index(feature)]
    quality = QualityMeasurerLinear()

    for p, a in zip(predicted, actual):
        quality.append(p[0], a[0])
    return quality
