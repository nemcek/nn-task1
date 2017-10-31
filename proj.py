"""
proj.py
"""

#!/usr/bin/env python
import random
import datetime
import csv
import os
from data import *
from classifier import *
from func import *
from functools import reduce
from experiments import *

# ARCHITECTURE = {
#     'outputs' : 3,
#     'layers' : [
#         {
#             'neurons': 2,
#             'function': 'tanh',
#             'weights': {
#                 'distribution': 'uniform',
#                 'scale': [0, 1]
#             }
#         },
#         {
#             'neurons': 20,
#             'function': 'logsig',
#             'weights': {
#                 'distribution': 'gauss',
#                 'scale': [0, 1]
#             }
#         },
#         {
#             'neurons': 6,
#             'function': 'linear',
#             'weights': {
#                 'distribution': 'uniform',
#                 'scale': [0, 1]
#             }
#         }
#     ],
# }

# TRAINING_SETTINGS = {
#     'learning_rate': 0.05,
#     'max_epochs': 200,
#     'momentum': 0.1,
#     'early_stopping': {
#         'min_accuracy': 97,
#         'best_weights_delay': {
#             'delay': 30
#         },
#         'raised_error': {
#             'threshold': 0.65,
#             'n_previous_epochs': 30
#         },
#         'accumulated_error': {
#             'threshold': 0.003,
#             'n_previous_epochs': 30
#         }
#     }
# }



def n_fold_cross_valid(d, n, architecture, training_settings):
    REs = []
    CEs = []
    epochs = []

    indices = np.arange(data.train_count)
    random.shuffle(indices)
    indices = np.split(indices, n)
    model = MLPClassifier(architecture)

    for i in range(n):
        split_indices = indices
        validation_inputs = data.train_inputs[:, split_indices[i]]
        validation_labels = data.train_labels[split_indices[i]]
        split_indices = np.concatenate(np.delete(split_indices, i, 0))
        estimation_inputs = data.train_inputs[:, split_indices]
        estimation_labels = data.train_labels[split_indices]

        model = MLPClassifier(architecture)
        _, _, last_epoch = model.train(estimation_inputs, estimation_labels, validation_inputs, validation_labels, training_settings, trace=False)
        testCE, testRE = model.test(validation_inputs, validation_labels)
        CEs.append(testCE)
        REs.append(testRE)
        epochs.append(last_epoch)

    return model, CEs, REs, epochs
    

if __name__ == '__main__':
    splits = 10
    data = Data()
    data.load_data('data/train.dat', 'data/test.dat', splits)

    now = datetime.datetime.now()
    file_name = 'results_{:4d}{:2d}{:2d}_{:2d}{:2d}.csv'.format(now.year, now.month, now.day, now.hour, now.minute)
    fieldnames = ['id', 'layers', 'functions', 'weights', 'scale', 'learning_rate', 'mean_epochs', 'momentum', 'min_accuracy', 'best_weights_delay', 'raised_error_threshold', 'raised_error_n', 'accumulated_error_threshold',
                    'accumulated_error_n', 'mean_CE', 'mean_RE', 'mean_n_epochs', 'architecture', 'training_settings']
    csvWriter = csv.DictWriter(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name), 'w', newline=''), fieldnames=fieldnames, delimiter=';')
    csvWriter.writeheader()

    for experiment in EXPERIMENTS:
        model, CEs, REs, epochs = n_fold_cross_valid(data, splits, experiment['architecture'], experiment['training_settings'])

        layers = [layer['neurons'] for layer in model.architecture['layers']]
        layers.append(model.architecture['outputs'])
        row = {
            'id': experiment['id'],
            'layers': layers,
            'functions': model.functions,
            'weights': [layer['weights']['distribution'] for layer in model.architecture['layers']],
            'scale': [layer['weights']['scale'] for layer in model.architecture['layers']],
            'learning_rate': model.training_settings['learning_rate'],
            'mean_epochs': model.training_settings['max_epochs'],
            'momentum': model.training_settings['momentum'],
            'min_accuracy': model.training_settings['early_stopping']['min_accuracy'],
            'best_weights_delay': model.training_settings['early_stopping']['best_weights_delay']['delay'],
            'raised_error_threshold': model.training_settings['early_stopping']['raised_error']['threshold'],
            'raised_error_n': model.training_settings['early_stopping']['raised_error']['n_previous_epochs'],
            'accumulated_error_threshold': model.training_settings['early_stopping']['accumulated_error']['threshold'],
            'accumulated_error_n': model.training_settings['early_stopping']['accumulated_error']['n_previous_epochs'],
            'mean_CE': np.mean(CEs),
            'mean_RE': np.mean(REs),
            'mean_epochs': np.mean(epochs),
            'architecture': model.architecture,
            'training_settings': model.training_settings
        }
        csvWriter.writerow(row)