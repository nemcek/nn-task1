import numpy as np
import queue as q

from mlp import *
from util import *
from func import *
from early_stopping import *

class MLPClassifier(MLP):

    def __init__(self, architecture):
        self.n_classes = architecture['outputs']
        super().__init__(architecture)

    ## prediction pass
    def predict(self, inputs):
        _, _, outputs = self.forward(inputs)
        return onehot_decode(outputs[-1])


    ## testing pass

    def test(self, inputs, labels):
        _, _, outputs = self.forward(inputs)
        targets = onehot_encode(labels, self.n_classes)
        predicted = onehot_decode(outputs)
        CE = np.sum(labels != predicted) / inputs.shape[1] 
        RE = np.sum(cost(targets, outputs)) / inputs.shape[1]
        return CE, RE


    def __init(self, training_settings):
       self.epochs = training_settings['max_epochs'] 
       self.alpha = training_settings['learning_rate']
       self.early_stopping = EarlyStopping(training_settings['early_stopping'], self.weights)
       self.momentum = training_settings['momentum']
       self.last_d_weights = list((np.zeros((weight.shape[0], weight.shape[1])).T for weight in self.weights))

    ## training
    def train(self, inputs, labels, validation_inputs, validation_labels, training_settings, trace=False, trace_interval=10):
        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.n_classes)
        self.__init(training_settings)

        if trace:
            ion()

        CEs = []
        REs = []

        for ep in range(self.epochs):
            print('Ep {:3d}/{}: '.format(ep+1, self.epochs), end='')
            CE = 0
            RE = 0

            for i in np.random.permutation(count):
                x = inputs[:, i]
                d = targets[:, i]

                y, dWeights = self.backward(x, d)

                CE += labels[i] != onehot_decode(y)
                RE += cost(d,y)

                for i in range(self.n_weights):
                    self.weights[i] += (self.alpha * dWeights[i].T) + (self.momentum * self.last_d_weights[i].T)

                self.last_d_weights = dWeights

            CE /= count
            RE /= count

            CEs.append(CE)
            REs.append(RE)

            if trace and ((ep+1) % trace_interval == 0):
                clear()
                predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                redraw()

            valid_ce, _ = self.test(validation_inputs, validation_labels)
            if (self.early_stopping.should_stop(ep, valid_ce, self.weights)):
                self.weights = self.early_stopping.weights
                print('validation error: {:6.2%}'.format(valid_ce))
                print('CE = {:6.2%}, RE = {:.5f}, vCE = {:6.2%}'.format(CE, RE, valid_ce))
                break

            print('CE = {:6.2%}, RE = {:.5f}, vCE = {:6.2%}'.format(CE, RE, valid_ce))

        if trace:
            ioff()

        print()

        return CEs, REs

    