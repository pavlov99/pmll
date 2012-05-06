import numpy as np
import unittest

class LinearRegressionLeastSquaresModel(object):
    def __init__(self, weights=None):
        self.weights = weights

    def train(self, objects):
        pass

    @staticmethod
    def get_weights(objects, labels):
        """
        w = (X' * X) ^ {-1} * X' * y
        """
        pass

    def get_regression_residuals(objects, labels):
        """
        y - Xw = (I - X * (X' * X) ^ {-1} * X') * y
        """
        pass


class LinearRegression(object):
    @staticmethod
    def get_regression(objects, weights):
        """
        Ret linear regression
        Input:
            objects - matrix (also X), representation of objects
            weights - vector (also w)

        Output:
            X * w
        """
        return np.asmatrix(objects) * weights

    @classmethod
    def get_regression1(cls, objects, weights):
        """
        Return np.matrix objects

        Output:
            [X, 1] * w
       """
        objects = np.asmatrix(objects)
        modified_objects = np.column_stack((
                objects,
                np.ones([objects.shape[0], 1]),
                ))
        return cls.get_regression(modified_objects, weights)


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

    def __init_weights(self, size):
        pass

    def train(self, objects, labels, max_iterations=100, accuracy=1e-5,
              regularization_parameter=1e-5):
        objects = np.hstack([
                objects,
                np.ones([objects.shape[0], 1]),
                ])
        labels = np.asmatrix(labels)


class IrlsClassifier(object):
    """
    Iterative reweighted least squares classifier.
    """
    def __init__(self, model):
        self.model = model

    def classify(self, objects):
        return LinearRegression.get_regression1(objects, self.model.weights)


class TestLinearRegressionLeastSquaresModel(unittest.TestCase):
    def setUp(self):
        pass


class TestLinearRegression(unittest.TestCase):
    def test_get_regression1_array_array(self):
        objects = np.random.rand(3,2)
        weights = np.random.rand(3,1)
        output = LinearRegression.get_regression1(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

    def test_get_regression1_array_matrix(self):
        objects = np.random.rand(3,2)
        weights = np.matrix(np.random.rand(3,1))
        output = LinearRegression.get_regression1(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

    def test_get_regression1_array_list(self):
        objects = np.random.rand(3,2)
        # list of arrays
        weights = list(np.random.rand(3,1))
        output = LinearRegression.get_regression1(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

        # list. Note, that it has to be list of lists: [[1], [2]] = column
        weights = [[float(w)] for w in weights]
        output = LinearRegression.get_regression1(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

    def test_get_regression1_matrix_array(self):
        objects = np.matrix(np.random.rand(3,2))
        weights = np.random.rand(3,1)
        output = LinearRegression.get_regression1(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

class TestIrlsModel(unittest.TestCase):
    def test__logit(self):
        logit = lambda x: IrlsModel._IrlsModel__logit(x)
        self.assertEqual(logit(0), 0.5)



class TestIrlsClassifier(unittest.TestCase):
    def setUp(self):
        pass

    def test_classify_existing_model(self):
        objects = [[0], [1]]
        weights = [[1], [2]]
        predicted_labels = np.matrix([[2], [3]])
        model = IrlsModel(weights)
        classifier = IrlsClassifier(model)
        labels = classifier.classify(objects)
        self.assertIsInstance(labels, np.matrix)
        self.assertEqual(labels.all(), predicted_labels.all())

if __name__ == '__main__':
    unittest.main()
