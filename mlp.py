import numpy as np

from util import *
from func import *

class MLP():

    def __init__(self, settings):
        self.dims       = [x.get('neurons') for x in settings['layers']]
        self.dims.append(settings['outputs'])
        self.layers     = len(self.dims)     # last dimension is output
        self.weights    = [np.random.rand(self.dims[i + 1], self.dims[i] + 1) for i in range(self.layers - 1)]
        self.settings   = settings

    def forward(self, x):
        layerInputs = []
        layerOutputs = []
        x = augment(x)

        # handle last one specifically
        for i in range(self.layers - 2):
            inputs, outputs = self.__forward(x, i)
            x = outputs
            layerInputs.append(inputs)
            layerOutputs.append(outputs)
        
        inputs, outputs = self.__forward(x, self.layers - 2)
        layerInputs.append(inputs)
        layerOutputs.append(outputs[: -1])  # without bias
        return layerInputs, layerOutputs, layerOutputs[-1]

    def __forward(self, x, i):
        a = self.weights[i] @ x
        h = augment(call(self.settings['layers'][i]['function'], a))
        return a, h


    def backward(self, x, d):
        layerInputs, layerOutputs, y = self.forward(x)
        dWeights = []
        g = None

        for i in reversed(range(self.layers - 1)):
            if i == self.layers - 2:    # last
                g = (d - y) * call(self.settings['layers'][i]['function'], layerInputs[i], isDerivation=True)
                dW = layerOutputs[i - 1].reshape((1, -1)).T * g.reshape((1, -1))
                dWeights.append(dW)
            elif i == 0:    # input
                g = (g.T @ self.weights[i + 1][:, 0:-1]) * call(self.settings['layers'][i]['function'], layerInputs[i], isDerivation=True)
                dW = augment(x).reshape((1, -1)).T * g.reshape((1, -1))
                dWeights.append(dW)
            else:
                g = (g.T @ self.weights[i + 1][:, 0:-1]) * call(self.settings['layers'][i]['function'], layerInputs[i], isDerivation=True)
                dW = layerOutputs[i - 1].reshape((1, -1)).T * g.reshape((1, -1))
                dWeights.append(dW)

        return y, list(reversed(dWeights))
