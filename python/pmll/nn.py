#!/usr/bin/python
# -*- coding: utf-8 -*-
from base import BaseModel, BaseClassifier
import numpy as np
import scipy
from scipy.io import loadmat
import unittest

class NeuralNet(BaseModel):
    @classmethod
    def _logit(cls, z, prime=0):
        """
        Activation function of neuron net. Here is used one function for each
        neurons -- sigmoid (logit).
        z - given point.
        prime - derivative order. By default function returns function value.
        """
        if prime == 0:
            return 1 / (1 + np.exp(-z))
        else:
            return cls._logit(z, prime - 1) * (1 - cls._logit(z, prime - 1))

    def __init__(self, hidden_neurons_sizes=None, activate=None):
        self.__hidden_heurons_sizes = hidden_neurons_sizes or []
        self.activate = activate or self._logit

    def __push_forward(self, x):
        """
        Push forward step, yield output after each layer
        """
        for w in self.weights:
            x = self.activate(x * w)
            yield x

    def _push_forward(self, x):
        return list(self.__push_forward(x))[-1]

    @staticmethod
    def _get_quality(predictions, labels):
        """
        return quality function for classification problem
        0 <= predictions <= 1

        Quality function of neural net.
        J = -(y==k)' * log(output[:,k]) - (y!=k)' * log(1 - output[:,k])
        labels here are already ones or zeros
        """
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        quality = -sum([np.dot(l, np.log(p)) + np.dot(1 - l, np.log(1 - p))
                        for p, l in zip(predictions.T, labels.T)])
        return quality

    def train(self, x, y):
        self.x = x
        self.y = y
        self.weights = [np.empty(shape) for shape in
                        zip([self.x.shape[1]] + self.__hidden_heurons_sizes,
                            self.__hidden_heurons_sizes + [self.y.shape[1]])]

        # print self.weights

    @property
    def predictor(self):
        return None

    def classify(self, x):
        print self._push_forward(x), self._push_forward(x).argmax(1)
        return self._push_forward(x).argmax(1)

class Perceptron(object):
    def train(self, x, y):
        """
        Realization of base train algorithm or linear perseptron
        """
        pass


class PerceptronClassifier(BaseClassifier):
    """
    Two classes classification
    """
    def __init__(self, weights):
        self.__weights = weights

    def classify(self, x):
        pass


class NeuralNetTest(unittest.TestCase):
    def setUp(self):
        """
         1
        0
        0101
        """
        self.x = np.matrix([
            [0, 0],
            [0, 1],
            [2, 0],
            [1, 2],
            [2, 1],
            [1, 0],
            ])

        # y represents indicators of classes
        self.y = np.matrix([
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 0],
                ])

        self.nn = NeuralNet()
        # print self.x
        # print self.y

    def test_train(self):
        model = self.nn.train(self.x, self.y)

    def test__push_forward(self):
        params = {"hidden_neurons_sizes": [4, 5]}
        nn = NeuralNet(**params)
        nn.train(self.x, self.y)
        outputs = list(nn._NeuralNet__push_forward(self.x))
        self.assertEqual(len(outputs), len(params["hidden_neurons_sizes"]) + 1)

        # check shapes, outputs for objects
        for output in outputs:
            self.assertEqual(output.shape[0], self.x.shape[0])

        self.assertEqual(outputs[-1].shape[1], self.y.shape[1])

    def test_push_forward(self):
        nn = NeuralNet()
        nn.train(self.x, self.y)
        output = nn._push_forward(self.x)
        self.assertEqual(output.shape, self.y.shape)

    def test_get_quality(self):
        self.nn.train(self.x, self.y)
        predictions = np.random.rand(*self.y.shape)
        q = self.nn._get_quality(predictions, self.y)
        self.assertTrue(isinstance(q, float))

    @unittest.skip("modify later")
    def test_classify(self):
        self.nn.train(self.x, self.y)
        output = self.nn.classify(self.x)
        self.assertEqual(output.shape, self.y.shape)


if __name__ == "__main__":
    unittest.main()
