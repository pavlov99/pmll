# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty
import time

from . import six


@six.add_metaclass(ABCMeta)
class BaseModel(object):
    """ Based model for algorithms.

    Each classifier or regression model should be derived from it.
    Useage:

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

    def __init__(self):
        self.__decorate_method('train')

    def __decorate_method(self, method_name):
        """Decorate given method (change to decoreted): calculate time."""
        method = getattr(self, method_name)

        def decorated_method(*args, **kwargs):
            # Decorate method: calculate time
            t = time.time()
            output = method(*args, **kwargs)
            setattr(self, '{0}_time'.format(method_name), time.time() - t)
            return output

        setattr(self, method_name, decorated_method)

    @abstractproperty
    def predictor(self):
        """ Return predictor for algorithm.

        Predictor property returns classifier or regressor or something
        which has call method and is able to predict labels for new objects.
        It is possible to initialize model and get predictor without train.
        This is useful case if you already know model parameters.

        Also predictor is a way to let model know how to deal with it (predict
        future answers based on parameters)

        """
        pass

    @abstractmethod
    def train(self, x, y):
        """Abstract train method, return self instance.

        Model fits parameters during train. It is possible to call
        self.predictor (for existing model) or self.train().predictor
        :return self:

        """
        return self


@six.add_metaclass(ABCMeta)
class BaseClassifier(object):

    """ Base class for any classification algorithm."""

    @abstractmethod
    def classify(self, x):
        pass

    def __call__(self, *args, **kwargs):
        self.classify(*args, **kwargs)


@six.add_metaclass(ABCMeta)
class BaseRegressor(object):

    """ Base class for any regression algorithm."""

    @abstractmethod
    def regress(self, x):
        pass

    def __call__(self, *args, **kwargs):
        self.regress(*args, **kwargs)
