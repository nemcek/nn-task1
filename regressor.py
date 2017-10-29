import numpy as np

from mlp import *
from util import *


class MLPRegressor(MLP):

    def __init__(self, dim_in, dim_hid, dim_out):
        super().__init__(dim_in, dim_hid, dim_out)

    
    ## functions

    def cost(self, targets, outputs):
        return np.sum((targets - outputs)**2, axis=0)

    def f_hid(self, X):
        return 1 / (1 + (np.e ** (-X)))

    def df_hid(self, X):
        return self.f_hid(X) * (1 - self.f_hid(X))

    def f_out(self, X):
        return X

    def df_out(self, X):
        return np.ones(len(X))


    ## training

    def train(self, inputs, targets, alpha=0.1, eps=100, trace=False):
        (_, count) = inputs.shape

        errors = []

        for ep in range(eps):
            print('Ep {:3d}/{}: '.format(ep+1, eps), end='')
            E = 0

            for i in np.random.permutation(count):
                x = inputs[:, i]
                d = targets[:, i]

                y, dW_hid, dW_out = self.backward(x, d)

                E += self.cost(d,y)

                self.W_hid += alpha * dW_hid.T
                self.W_out += alpha * dW_out.T

            E /= count
            errors.append(E)
            print('E = {:.3f}'.format(E))

        return errors
