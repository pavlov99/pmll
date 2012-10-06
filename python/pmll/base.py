# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty
import time
import random
import unittest

__author__  = "Kirill Pavlov"
__email__   = "kirill.pavlov@phystech.edu"


def timer():
    def wrapper(f):
        def wrapped_function(*args, **kwargs):
            t = time.time()
            output = f(*args, **kwargs)
            print time.time() - t
            return output


class BaseModel(object):
    """
    Based model for algorithms. Each classifier or regression model should be
    derived from it. Common useage is:

    class Model(BaseModel):
        @property
        def predictor(self): pass

        def train(self, objects, labels): return self

    model = Model()

    model.predictor(objects_test) -> labels_test
    model.train(objects_train, labels_train).predictor(objects_test)
      -> labels_test

    or create predictor direcly:
    predict = model.predictor
    predict(objects_test) -> labels_test

    predictor property returns objects which can predict labels for new objects
    it is also possible to call corresponding method for each class:
    Classifier = ClassifierModel().predictor
    Classifier.classify(object_test) -> label_test
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.__decorate_method('train')

    def __decorate_method(self, method_name):
        """
        Decorate given method (change to decoreted):
          - calculate time.
        """
        method = getattr(self, method_name)

        def decorated_method(*args, **kwargs):
            # Decorate method: calculate time
            t = time.time()
            output = method(*args, **kwargs)
            setattr(self, '%s_time' % method_name, time.time() - t)
            return output

        setattr(self, method_name, decorated_method)

    @abstractproperty
    def predictor(self):
        """
        predictor property returns classifier or regressor or something
        which has call method and is able to predict labels for new objects.
        It is possible to initialize model and get predictor without train.
        This is useful case if you already know model parameters.

        Also predictor is a way to let model know how to deal with it (predict
        future answers based on parameters)
        """
        pass

    @abstractmethod
    def train(self, x, y):
        """
        Abstract train method, return self instance. It is possible to call
        self.predictor or self.train().predictor
        """
        return self


class BaseClassifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def classify(self, x):
        pass

    def __call__(self, *args, **kwargs):
        self.classify(*args, **kwargs)


class BaseRegressor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def regress(self, x):
        pass

    def __call__(self, *args, **kwargs):
        self.regress(*args, **kwargs)


class BaseModelTest(unittest.TestCase):
    def setUp(self):
        class Model(BaseModel):
            @property
            def predictor(self):
                pass

            def train(self):
                return self

        self.class_ = Model
        self.model = self.class_()

    def test_basemodel_require_train(self):
        BadModel = self.class_
        delattr(BadModel, 'train')
        self.assertRaises(BadModel())

    def test_basemodel_require_predictor(self):
        BadModel = self.class_
        delattr(BadModel, 'predictor')
        self.assertRaises(BadModel())

    def test_basemodel_adds_time_to_train(self):
        self.assertFalse(hasattr(self.model, 'train_time'))
        self.model.train()
        self.assertTrue(hasattr(self.model, 'train_time'))


class BaseClassifierTest(unittest.TestCase):
    def setUp(self):
        class Classifier(BaseClassifier):
            @property
            def classify(self):
                return self

        self.class_ = Classifier
        self.model = self.class_()

    def test_require_classify(self):
        BadModel = self.class_
        delattr(BadModel, 'classify')
        self.assertRaises(BadModel())

    def test_is_callable(self):
        # it is possible to call model
        self.assertTrue(hasattr(self.model, '__call__'))

        # () is link to classify
        self.assertEqual(self.model, self.model.classify)


class BaseRegressorTest(unittest.TestCase):
    def setUp(self):
        class Regressor(BaseRegressor):
            @property
            def regress(self):
                return self

        self.class_ = Regressor
        self.model = self.class_()

    def test_require_regress(self):
        BadModel = self.class_
        delattr(BadModel, 'regress')
        self.assertRaises(BadModel())

    def test_is_callable(self):
        # it is possible to call model
        self.assertTrue(hasattr(self.model, '__call__'))

        # () is link to regress
        self.assertEqual(self.model, self.model.regress)


if __name__ == '__main__':
    unittest.main()
