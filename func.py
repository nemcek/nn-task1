import numpy as np

## general activations

def relu(X):
    return np.where(X>0, X, 0)

def drelu(X):
    return np.where(X>0, 1, 0)


def logsig(X):
    return 1 / (1 + np.exp(-X))

def dlogsig(X):
    return logsig(X) * (1 - logsig(X))


def tanh(X):
    return np.tanh(X)

def dtanh(X):
    return 1 / np.cosh(X)**2


## output-only activations

def linear(X):
    return X

def dlinear(X):
    return np.ones(X.shape)


def softmax(X):
    # X = np.atleast_2d(X)
    # return (np.exp(X) / np.sum(np.exp(X), axis=1))
    return (np.exp(X) / np.sum(np.exp(X)))

def dsoftmax(X):
    # X = np.atleast_2d(X)
    # soft = np.exp(X) / np.sum(np.exp(X), axis=1)
    # out = np.zeros(X.shape)
    # out[range(X.shape[0]), np.argmax(X, axis=1)] = soft[range(X.shape[0]), np.argmax(X, axis=1)]
    # return out
    out = np.zeros(X.shape)
    out[np.argmax(X)] = np.exp(X[np.argmax(X)]) / np.sum(np.exp(X))
    return out


## cost functions

def cost(targets, outputs): # new
        return np.sum((targets - outputs)**2, axis=0)

def MSE(D, Y):
    return np.sum((D - Y)**2, axis=0)

# TODO dMSE


def CE(D, Y):
    return -np.sum(np.log(Y) * D, axis=0)

# TODO dCE

FUNCTIONS = {
    'relu' : { 'func': relu, 'dfunc': drelu },
    'logsig' : { 'func': logsig, 'dfunc': dlogsig },
    'tanh' : { 'func': tanh, 'dfunc': dtanh },
    'softmax' : { 'func': softmax, 'dfunc': dsoftmax },
    'linear' : { 'func': linear, 'dfunc': dlinear }
}

def call(func, val, isDerivation=False):
    f = FUNCTIONS.get(func)
    if (isDerivation):
        return f.get('dfunc')(val)
    else:
        return f.get('func')(val)