#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from neuralnet import NeuralLayer, NeuralNet

class NeuralLayerTestCase(unittest.TestCase):
    objects = np.matrix([[0, 1, 2], [-1, 0, 0]])

    def setUp(self):
        self.layer = NeuralLayer(3, 1)

    def test_push_forward_base(self):
        self.layer.push_forward(NeuralLayerTestCase.objects)

    def test_init_initial_weights(self):
        initial_weights = np.asarray([[1], [2], [3]])
        layer = NeuralLayer(3, 1, initial_weights=initial_weights)
        self.assertEqual(layer.number_inputs, 3)
        self.assertEqual(list(initial_weights.flat),
                         list(layer.weights.flat))

        layer = NeuralLayer(2, 1, initial_weights=initial_weights,
                            constant_neuron=True)
        self.assertEqual(layer.number_inputs, 3)
        self.assertEqual(list(initial_weights.flat),
                         list(layer.weights.flat))

        layer = NeuralLayer(3, 1, initial_weights=initial_weights,
                            constant_neuron=True)
        self.assertEqual(layer.number_inputs, 4)
        self.assertEqual(list(initial_weights.flat), list(layer.weights.flat))
        

if __name__ == '__main__':
    unittest.main()
