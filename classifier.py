import numpy as np

from mlp import *
from util import *
from func import *

class MLPClassifier(MLP):

    def __init__(self, settings):
        self.n_classes = settings['outputs']
        super().__init__(settings)

    ## prediction pass
    def predict(self, inputs):
        # outputs, *_ = self.forward(inputs)  # if self.forward() can take a whole batch
        # outputs = np.stack([self.forward(x)[0] for x in inputs.T]) # otherwise
        _, _, outputs = self.forward(inputs)
        # return onehot_decode(outputs)
        return onehot_decode(outputs[-1])


    ## testing pass

    def test(self, inputs, labels):
        # outputs, *_ = self.forward(inputs)
        _, _, outputs = self.forward(inputs)
        targets = onehot_encode(labels, self.n_classes)
        predicted = onehot_decode(outputs)
        CE = np.sum(labels != predicted) / inputs.shape[1] 
        RE = np.sum(cost(targets, outputs)) / inputs.shape[1]
        return CE, RE


    ## training

    def train(self, inputs, labels, alpha=0.1, eps=100, trace=False, trace_interval=10):
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.n_classes)

        if trace:
            ion()

        CEs = []
        REs = []

        for ep in range(eps):
            print('Ep {:3d}/{}: '.format(ep+1, eps), end='')
            CE = 0
            RE = 0

            for i in np.random.permutation(count):
                x = inputs[:, i]
                d = targets[:, i]

                y, dWeights = self.backward(x, d)

                CE += labels[i] != onehot_decode(y)
                RE += cost(d,y)

                for i in range(self.layers - 1):
                    self.weights[i] += alpha * dWeights[i].T
                # self.W_hid += alpha * dW_hid.T
                # self.W_out += alpha * dW_out.T

            CE /= count
            RE /= count

            CEs.append(CE)
            REs.append(RE)

            print('CE = {:6.2%}, RE = {:.5f}'.format(CE, RE))

            if trace and ((ep+1) % trace_interval == 0):
                clear()
                predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                redraw()

        if trace:
            ioff()

        print()

        return CEs, REs
