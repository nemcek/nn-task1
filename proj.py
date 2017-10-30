"""
proj.py
"""

#!/usr/bin/env python
import random
from data import *
from classifier import *
from func import *
from functools import reduce


ARCHITECTURE = {
    'outputs' : 3,
    'layers' : [
        {
            'neurons': 2,
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

TRAINING_SETTINGS = {
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

# model = MLPClassifier(architecture)
# trainCEs, trainREs = model.train(data.estimation_inputs, data.estimation_labels, data.validation_inputs, data.validation_labels, training_settings, trace=False, trace_interval=10)
 
# testCE, testRE = model.test(data.validation_inputs, data.validation_labels)
# print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))
# 
# plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False) 

def n_fold_cross_valid(d, n):
    REs = []
    CEs = []
    model = MLPClassifier(ARCHITECTURE)

    print(data.train_inputs)
    print(data.train_labels)
    indices = np.arange(data.train_count)
    random.shuffle(indices)
    indices = np.split(indices, n)
    for i in range(n):
        split_indices = indices
        validation_inputs = data.train_inputs[:, split_indices[i]]
        validation_labels = data.train_labels[split_indices[i]]
        split_indices = np.concatenate(np.delete(split_indices, i, 0))
        estimation_inputs = data.train_inputs[:, split_indices]
        estimation_labels = data.train_labels[split_indices]
        print(estimation_inputs)
        print(estimation_labels)

        trainCEs, trainREs = model.train(estimation_inputs, estimation_labels, validation_inputs, validation_labels, TRAINING_SETTINGS, trace=False)

if __name__ == '__main__':
    splits = 10
    data = Data()
    data.load_data('data/train.dat', 'data/test.dat', splits)
    n_fold_cross_valid(data, splits)