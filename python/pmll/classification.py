import numpy as np
import unittest

class LinearRegressionLeastSquaresModel(object):
    def __init__(self, weights=None):
        self.weights = weights

    def train(self, objects):
        pass

    @staticmethod
    def get_weights(objects, labels, object_weights=None):
        """
        w = (X' * W * X) ^ {-1} * X' * W * y
        """
        X = np.asmatrix(objects)
        y = np.asmatrix(labels)
        if object_weights is None:
            return (X.T * X).I * X.T * y
        else:
            W = np.diagflat(object_weights)
            return (X.T * W * X).I * X.T * W * y

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

    @staticmethod
    def __logit(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def __get_weights(size, min_weight=None, max_weight=None):
        """
        Return initial values for weights. By default w \in [-1/2n, 1/2n]
        where n = size. It is possible to define min and max values for
        weights
        """
        a = min_weight or -0.5 / size
        b = max_weight or 0.5 / size
        return np.matrix((b - a) * np.random.random_sample((size, 1)) + a)

    def __init_weights(self, size):
        pass

    def train(self, objects, labels, max_iterations=100, accuracy=1e-5,
              regularization_parameter=1e-5):

        # change types of lobjects and labels to np.matrix
        objects = np.hstack([
                objects,
                np.ones([objects.shape[0], 1]),
                ])
        labels = np.asmatrix(labels)

        # Initialize weights
        self.weights = self.weights or self.__get_weights(objects.shape[1])
        self.__history = {'weights': self.weights}


class IrlsClassifier(object):
    """
    Iterative reweighted least squares classifier.
    """
    def __init__(self, model):
        self.model = model

    def classify(self, objects):
        weights = self.model.weights or\
            IrlsModel._IrlsModel__get_weights(np.asmatrix(objects).shape[1] + 1)
        return LinearRegression.get_regression1(objects, weights)


class TestLinearRegressionLeastSquaresModel(unittest.TestCase):
    def setUp(self):
        nobjects, nfeatures = 3, 5
        self.objects = np.random.rand(3,2)
        self.labels = [[0], [1], [2]]
        self.object_weights = [[1], [0], [0]]
        self.feature_weights = [[0], [1]]

    def test_get_weights(self):
        weights = LinearRegressionLeastSquaresModel.get_weights(
            self.objects, self.labels)
        self.assertIsInstance(weights, np.matrix)
        self.assertEqual(weights.shape, (2, 1))

    def test_get_weights_objects(self):
        weights = LinearRegressionLeastSquaresModel.get_weights(
            self.objects, self.labels, self.object_weights)
        self.assertIsInstance(weights, np.matrix)
        self.assertEqual(weights.shape, (2, 1))
        self.assertTrue((weights == np.matrix([[0], [0]])).all())


class TestLinearRegression(unittest.TestCase):
    def test_get_regression1_array_array(self):
        objects = np.random.rand(3,2)
        weights = np.random.rand(3,1)
        output = LinearRegression.get_regression1(objects, weights)
        self.assertIsInstance(output, np.matrix)
        self.assertEqual(output.shape, (3, 1))

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

    def test__get_weights(self):
        size = np.random.randint(1, high=100)
        min_weight, max_weight = sorted(np.random.rand(2))

        weights = IrlsModel._IrlsModel__get_weights(
            size, min_weight, max_weight)

        self.assertIsInstance(weights, np.matrix)
        self.assertTrue((min_weight <= weights).all())
        self.assertTrue((weights < max_weight).all())
        self.assertEqual(weights.shape, (size, 1))

    def test__get_weights_default(self):
        size = 1000
        min_weight, max_weight = -0.5 / size, 0.5 / size
        weights = IrlsModel._IrlsModel__get_weights(size)

        self.assertIsInstance(weights, np.matrix)
        self.assertTrue((min_weight <= weights).all())
        self.assertTrue((weights < max_weight).all())
        self.assertEqual(weights.shape, (size, 1))


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

    def test_classify_default_model(self):
        objects = [[0], [1]]
        model = IrlsModel()
        classifier = IrlsClassifier(model)
        labels = classifier.classify(objects)
        self.assertIsInstance(labels, np.matrix)


if __name__ == '__main__':
    unittest.main()
