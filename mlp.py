import numpy as np

from util import *
from func import *
from data import DataOperations

class MLP():

    def __init__(self, architecture):
        self.__build(architecture)

    def __build(self, architecture):
        dims = [layer['neurons'] for layer in architecture['layers']]
        dims.append(architecture['outputs'])

        self.weights    = [DataOperations.generate_distribution(
            layer['weights']['distribution'], 
            [dims[i + 1], dims[i] + 1], 
            layer['weights']['scale']) for i, layer in enumerate(architecture['layers'])]
        self.n_weights  = len(self.weights)
        self.functions  = [layer['function'] for layer in architecture['layers']]
        self.architecture   = architecture

    def forward(self, x):
        layer_inputs = []
        layer_outputs = []
        x = augment(x)

        # handle last one specifically
        for i in range(self.n_weights - 1):
            inputs, outputs = self.__forward(x, i)
            x = outputs
            layer_inputs.append(inputs)
            layer_outputs.append(outputs)
        
        inputs, outputs = self.__forward(x, self.n_weights - 1)
        layer_inputs.append(inputs)
        layer_outputs.append(outputs[: -1])  # without bias
        return layer_inputs, layer_outputs, layer_outputs[-1]

    def __forward(self, x, i):
        inputs = self.weights[i] @ x
        outputs = augment(call(self.functions[i], inputs))
        return inputs, outputs


    def backward(self, x, d):
        layer_inputs, layer_outputs, y = self.forward(x)
        d_weights = []
        g = None

        for i in reversed(range(self.n_weights)):
            if (i == self.n_weights - 1):   # last
                g = (d - y)
                d_weight = layer_outputs[i - 1].reshape((1, -1)).T
            elif i == 0:                    # first
                g = (g.T @ self.weights[i + 1][:, 0:-1])
                d_weight = augment(x).reshape((1, -1)).T
            else:                           # between
                g = (g.T @ self.weights[i + 1][:, 0:-1])
                d_weight = layer_outputs[i - 1].reshape((1, -1)).T

            g = g * call(self.functions[i], layer_inputs[i], isDerivation=True)
            d_weight = d_weight * g.reshape((1, -1))
            d_weights.append(d_weight)

        return y, list(reversed(d_weights))
