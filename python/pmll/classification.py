import numpy as np
import unittest


class LinearRegressionLeastSquaresModel(object):
    def __init__(self, weights=None):
        self.weights = weights

    def train(self, objects, labels):
        pass

    @staticmethod
    def get_weights(objects, labels, object_weights=None, regularization=0):
        """
        w = (X' * W * X) ^ {-1} * X' * W * y
        """
        X = np.asmatrix(objects)
        y = np.asmatrix(labels)
        I = regularization * np.eye(objects.shape[1]) # FIXIT other reg-s ?
        if object_weights is None:
            return (X.T * X + I).I * X.T * y
        else:
            W = np.diagflat(object_weights)
            return (X.T * W * X + I).I * X.T * W * y

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
        # print "bjects", type(np.asmatrix(objects)), np.asmatrix(objects)
        # print "w", type(weights), weights
        return np.asmatrix(objects) * np.asmatrix(weights)

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

    @staticmethod
    def __get_change(vector1, vector2, norm="inf"):
        if norm == "inf":
            return float(sum(abs(vector1 - vector2)))
        if norm == "0":
            return abs(vector1 - vector2).max()

    def __is_stop(self, accuracy):
        return self.__history['weight_change'][-1] < accuracy

    def train(self, objects, labels, object_weights=None, max_iterations=100,
              accuracy=1e-5, regularization=1e-5):

        # change types of lobjects and labels to np.matrix
        # objects = wide objects [X, 1]
        objects = np.asmatrix(np.column_stack((
                objects,
                np.ones([objects.shape[0], 1]),
                )))
        labels = np.asmatrix(labels)
        if object_weights is None:
            object_weights = np.array([[1]] * objects.shape[0])
        I = regularization * np.eye(objects.shape[1])

        # Initialize weights
        self.weights = self.weights or self.__get_weights(objects.shape[1])
        self.__history = {'weights': [self.weights], 'weight_change': []}

        for iteration in range(max_iterations):
            classifier = IrlsClassifier(self)
            probability = classifier.classify(objects[:, :-1])
            object_weights_new = np.multiply(
                probability - np.power(probability, 2),
                object_weights,
                )

            X = objects
            y = np.asmatrix(probability - labels)
            W = np.diagflat(object_weights_new)

            self.weights = self.weights - (X.T * W * X + I).I * X.T * y
            self.__history['weights'].append(self.weights)
            self.__history['weight_change'].append(self.__get_change(
                    self.__history['weights'][-2],
                    self.__history['weights'][-1]))

            if self.__is_stop(accuracy):
                break


class IrlsClassifier(object):
    """
    Iterative reweighted least squares classifier.
    """
    def __init__(self, model):
        self.model = model

    def classify(self, objects):
        if self.model.weights is None:
            weights = IrlsModel._IrlsModel__get_weights(
                np.asmatrix(objects).shape[1] + 1)
        else:
            weights = self.model.weights

        logit = lambda z: IrlsModel._IrlsModel__logit(z)
        return logit(LinearRegression.get_regression1(objects, weights))


class ModelMixtureModel(object):
    def __init__(self, weights=None, number_models=None):
        self.weights = weights
        if weights is not None:
            self.number_models = weights.shape[1]

    def train(self, objects, labels, number_models, object_weights=None,
              max_iterations=100, accuracy=1e-5, regularization=1e-5):
        pass


class ModelMixtureClassifier(object):
    def __init__(self, model):
        self.model = model

    def classify(self, objects):
        pass


class TestLinearRegressionLeastSquaresModel(unittest.TestCase):
    def setUp(self):
        nobjects, nfeatures = 3, 5
        self.objects = np.random.rand(3, 2)
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

    def test_get_weights_lin_dependency(self):
        objects = np.matrix([[1, 1], [2, 2], [3, 3]])
        with self.assertRaises(np.linalg.LinAlgError):
            LinearRegressionLeastSquaresModel.get_weights(objects, self.labels)

        LinearRegressionLeastSquaresModel.get_weights(
            objects, self.labels, regularization=1)


class TestLinearRegression(unittest.TestCase):
    def test_get_regression1_array_array(self):
        objects = np.random.rand(3, 2)
        weights = np.random.rand(3, 1)
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
        objects = np.random.rand(3, 2)
        # list of arrays
        weights = list(np.random.rand(3, 1))
        output = LinearRegression.get_regression1(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

        # list. Note, that it has to be list of lists: [[1], [2]] = column
        weights = [[float(w)] for w in weights]
        output = LinearRegression.get_regression1(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)

    def test_get_regression1_matrix_array(self):
        objects = np.matrix(np.random.rand(3, 2))
        weights = np.random.rand(3, 1)
        output = LinearRegression.get_regression1(objects, weights)
        self.assertEqual(output.shape, (3, 1))
        self.assertIsInstance(output, np.matrix)


class TestIrlsModel(unittest.TestCase):
    def setUp(self):
        nobjects, nfeatures = 10, 2
        self.objects = np.random.rand(nobjects, nfeatures)
        self.labels = np.asmatrix(np.random.randint(0, 2, nobjects)).T

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

    def test_train(self):
        model = IrlsModel()
        model.train(self.objects, self.labels)


class TestIrlsClassifier(unittest.TestCase):
    def setUp(self):
        pass

    def test_classify_answer_interval(self):
        nobjects, nfeatures = 1000, 3
        objects = np.random.rand(nobjects, nfeatures)
        weights = np.random.rand(nfeatures + 1, 1)
        classifier = IrlsClassifier(IrlsModel(weights))
        labels = classifier.classify(objects)
        self.assertIsInstance(labels, np.matrix)
        self.assertTrue((labels <= 1).all())
        self.assertTrue((labels >= 0).all())

    def test_classify_existing_model(self):
        objects = [[0], [2]]
        weights = [[1], [-1]]
        model = IrlsModel(weights)
        classifier = IrlsClassifier(model)
        labels = classifier.classify(objects)
        self.assertIsInstance(labels, np.matrix)
        self.assertTrue(float(labels[0]) < 0.5)
        self.assertTrue(float(labels[1]) > 0.5)

    def test_classify_default_model(self):
        objects = [[0], [1]]
        model = IrlsModel()
        classifier = IrlsClassifier(model)
        labels = classifier.classify(objects)
        self.assertIsInstance(labels, np.matrix)


if __name__ == '__main__':
    unittest.main()
