#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from scipy.io import loadmat

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
        oblects = (number_objects, number_inputs) matrix
        """
        if self.constant_neuron:
            return np.hstack([objects, np.ones([objects.shape[0], 1])]) *\
                self.weights
        else:
            return objects * self.weights

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
    
    
    def __init__(self, number_neurons):
        """
        data - structure with "objects" and "labels" fields
        numberNeurons - number neurons in corresponding layer
        """
        if len(number_neurons) < 2:
            raise AssertionError("Number of layers must be larger or equal to 2. Input layer and output layer.")

        self.layers = [NeuralLayer(number_inputs, number_outputs) 
                       for number_inputs, number_outputs 
                       in zip(number_neurons[:-1], number_neurons[1:])]

    def push_forward(self, objects):
        pass

    def train(objects, labels):
        pass

    @classmethod
    def from_objects_labels(cls, objects, labels, number_hidden_neurons=[]):
        return cls([objects.shape[1]] + number_hidden_neurons + [labels.shape[1]])



if __name__ == '__main__':
    data = scipy.io.loadmat('../data/iris.mat')
    x = data['X'][50:]
    y = data['Y'][50:] - 1
    
    print 'hello'
