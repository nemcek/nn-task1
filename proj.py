"""
proj.py
"""

#!/usr/bin/env python
from data import *
from classifier import *
from func import *

data = Data()
data.load_data('data/train.dat', 'data/test.dat', 20)

architecture = {
    'outputs' : np.max(data.estimation_labels) + 1,
    'layers' : [
        {
            'neurons': data.dims,
            'function': 'tanh',
            'weights': {
                'distribution': 'uniform',
                'scale': [0, 1]
            }
        },
        {
            'neurons': 20,
            'function': 'logsig',
            'weights': {
                'distribution': 'gauss',
                'scale': [0, 1]
            }
        },
        {
            'neurons': 6,
            'function': 'linear',
            'weights': {
                'distribution': 'uniform',
                'scale': [0, 1]
            }
        }
    ],
}

training_settings = {
    'learning_rate': 0.05,
    'max_epochs': 200,
    'momentum': 0.1,
    'early_stopping': {
        'min_accuracy': 97,
        'best_weights_delay': {
            'delay': 50
        },
        'raised_error': {
            'threshold': 0.8,
            'n_previous_epochs': 30
        },
        'accumulated_error': {
            'threshold': 0.5,
            'n_previous_epochs': 30
        }
    }
}

model = MLPClassifier(architecture)
trainCEs, trainREs = model.train(data.estimation_inputs, data.estimation_labels, data.validation_inputs, data.validation_labels, training_settings, trace=False, trace_interval=10)
 
testCE, testRE = model.test(data.validation_inputs, data.validation_labels)
print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False) 