import numpy as np
import unittest

class LinearRegressionLeastSquaresModel(object):
    def __init__(self, weights=None):
        self.weights = weights

    def train(self, objects):
        pass


class LinearRegressionLeastSquares(object):
    @staticmethod
    def get_regression(objects, weights):
        """
        Return np.matrix objects
        """
        modified_objects = np.column_stack((
                objects,
                np.ones([objects.shape[0], 1]),
                ))
        return np.asmatrix(modified_objects) * weights

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


class TestLinearRegressionLeastSquaresModel(unittest.TestCase):
    def setUp(self):
        pass


class TestLinearRegressionLeastSquares(unittest.TestCase):
    def test_get_regression_array_array(self):
        objects = np.random.rand(3,2)
        weights = np.random.rand(3,1)
        output = LinearRegressionLeastSquares.get_regression(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

    def test_get_regression_array_matrix(self):
        objects = np.random.rand(3,2)
        weights = np.matrix(np.random.rand(3,1))
        output = LinearRegressionLeastSquares.get_regression(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

    def test_get_regression_array_list(self):
        objects = np.random.rand(3,2)
        # list of arrays
        weights = list(np.random.rand(3,1))
        output = LinearRegressionLeastSquares.get_regression(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

        # list. Note, that it has to be list of lists: [[1], [2]] = column
        weights = [[float(w)] for w in weights]
        output = LinearRegressionLeastSquares.get_regression(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

    def test_get_regression_matrix_array(self):
        objects = np.matrix(np.random.rand(3,2))
        weights = np.random.rand(3,1)
        output = LinearRegressionLeastSquares.get_regression(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

class TestIrlsModel(unittest.TestCase):
    def test__logit(self):
        logit = lambda x: IrlsModel._IrlsModel__logit(x)
        self.assertEqual(logit(0), 0.5)



class TestIrlsClassifier(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
