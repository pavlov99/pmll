#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from scipy import optimize
from scipy.io import loadmat
import itertools

class NeuralLayer(object):
    """
    Class represents one layer for neural net.
    """
    @staticmethod
    def __init_weights(number_inputs, number_outputs):
        return np.empty([number_inputs, number_outputs])

    def __init__(self, number_inputs, number_outputs, constant_neuron=False,
                 initial_weights=None):
        """
        constant_neuron - boolean indicator of neuron with constant output
        """
        self.constant_neuron = constant_neuron
        self.number_inputs = number_inputs if not constant_neuron else number_inputs + 1
        self.number_outputs = number_outputs
        if initial_weights is not None:
            initial_weights = np.asarray(initial_weights)
            if initial_weights.shape != (self.number_inputs,
                                         self.number_outputs):
                raise AssertionError("Shape of initial weight matrix %s and \
(number_inputs=%s, number_outputs=%s) must be equal." % (initial_weights.shape,
                                                         self.number_inputs,
                                                         self.number_outputs,
                                                         )
                                     )
            self.weights = initial_weights
        else:
            self.weights = self.__init_weights(number_inputs, number_outputs)

    def __repr__(self):
        return "%s: [%s x %s]" % (self.__class__,
                                  self.number_inputs,
                                  self.number_outputs)

    def push_forward(self, objects):
        """
        return output signal of layer given input signal.
        objects = (number_objects, number_inputs) matrix
        """
        if self.constant_neuron:
            return dot(np.hstack([objects, np.ones([objects.shape[0], 1])]),
                       self.weights)
        else:
            return np.dot(objects, self.weights)


class NeuralNet(object):
    @classmethod
    def activation_function(cls, z, prime=0):
        """
        Activation function of neuron net. Here is used one function for each
        neurons -- sigmoid (logit).
        z - given point.
        prime - derivative order. By default function returns function value.
        """
        if prime == 0:
            return 1 / (1 + np.exp(-z))
        else:
            return cls.activation_function(z, prime - 1) * \
                (1 - cls.activation_function(z, prime - 1))

    def quality_function(self, predictions, labels, regularization=0):
        """
        Quality function of neural net.
        J = -(y==k)' * log(output[:,k]) - (y!=k)' * log(1 - output[:,k])
        """
        quality = 0
        for index, label in enumerate(set(labels.flat)):
            quality = quality -\
                np.dot((labels == label).T, np.log(predictions[:, index])) -\
                np.dot((labels != label).T, np.log(1 - predictions[:, index]))
        quality = quality / labels.shape[0]
        return quality

    def __init__(self, number_neurons, constant_neuron=False):
        """
        data - structure with "objects" and "labels" fields
        numberNeurons - number neurons in corresponding layer
        """
        if len(number_neurons) < 2:
            raise AssertionError("Number of layers must be larger or equal to 2. Input layer and output layer.")

        self.layers = [NeuralLayer(number_inputs, number_outputs,
                                   constant_neuron=constant_neuron)
                       for number_inputs, number_outputs
                       in zip(number_neurons[:-1], number_neurons[1:])]

    def push_forward(self, objects):
        output = objects
        for layer in self.layers:
            output = self.activation_function(layer.push_forward(output))
            yield output

    @staticmethod
    def __normalize_output(matrix):
        """
        divide each row of matrix by sum of elements in this row
        """
        return matrix / np.tile(np.sum(matrix, 1)[:,np.newaxis],
                                (1, matrix.shape[1]))

    def get_quality(self, objects, labels, weights=None):
        """
        Return quality of neural net. Use quality function for classification.
        """
        if weights is not None:
            initial_weights = self.unroll_weights()
            self.roll_weights(weights)

        # Neural net output
        outputs = self.__normalize_output(list(self.push_forward(objects))[-1])

        if weights is not None:
            self.roll_weights(initial_weights)

        return self.quality_function(outputs, labels)

    def unroll_weights(self):
        """
        Return generator - vector of weights
        """
        return itertools.chain(*(layer.weights.flat for layer in self.layers))


    def roll_weights(self, weights):
        """
        Convert vector representation to weights and assign to layers
        """
        weights = list(weights)
        layer_sizes = [layer.weights.size for layer in self.layers]
        assert(len(weights) == sum(layer_sizes))

        layer_indexes = np.cumsum([0] + layer_sizes)
        for index, layer in enumerate(self.layers):
            new_weights = np.asarray(weights[layer_indexes[index]:\
                                                 layer_indexes[index + 1]])
            self.layers[index].weights = new_weights.reshape(layer.weights.shape)


    def learn_gradient(self, objects, labels, initial_weights=None):
        weights = initial_weights or list(self.unroll_weights())
        f = lambda w: self.get_quality(objects, labels, weights=w)
        weights_opt = optimize.fmin_bfgs(f, weights, maxiter=50, disp=False)
        self.roll_weights(weights_opt)

    def train(objects, labels):
        """
        Train neural net using back propogation
        """
        pass

    def classify(self, objects):
        return list(self.push_forward(objects))[-1].argmax(1)

    @classmethod
    def from_objects_labels(cls, objects, labels, number_hidden_neurons=[],
                            constant_neuron=False):
        number_inputs = objects.shape[1]
        if labels.shape[1] == 1:
            # labels = list of class labels
            number_outputs = len(set(labels.flat))
        else:
            # labels = matrix, each row is boolean indicator of class.
            # Line contains 1 in the column corresponding to the object class.
            number_outputs = labels.shape[1]
        return cls([number_inputs] + number_hidden_neurons + [number_outputs])



if __name__ == '__main__':
    data = scipy.io.loadmat('../data/iris.mat')
    x = data['X'][:]
    y = data['Y'][:]
    nn = NeuralNet.from_objects_labels(x, y)

    # print list(nn.unroll_weights())
    #print nn.get_quality(x, y)

    # print nn.roll_weights(range(12))
    # print list(nn.unroll_weights())
    #print nn.get_quality(x, y, range(12))

    # print nn._get_weight_gradient(x, y)
    nn.learn_gradient(x, y)
    # print list(nn.push_forward(x))[-1]
    print nn.classify(x)
