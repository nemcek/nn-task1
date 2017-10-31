EXPERIMENTS = [
    {
        'id': 1,
        'architecture': {
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
                        'distribution': 'uniform',
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
            ]
        },
        'training_settings': {
            'learning_rate': 0.05,
            'max_epochs': 200,
            'momentum': 0.1,
            'early_stopping': {
                'min_accuracy': 97,
                'best_weights_delay': {
                    'delay': 30
                },
                'raised_error': {
                    'threshold': 0.65,
                    'n_previous_epochs': 30
                },
                'accumulated_error': {
                    'threshold': 0.003,
                    'n_previous_epochs': 30
                }
            }
        }
    },
    {
        'id': 2,
        'architecture': {
            'outputs' : 3,
            'layers' : [
                {
                    'neurons': 2,
                    'function': 'tanh',
                    'weights': {
                        'distribution': 'gauss',
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
                        'distribution': 'gauss',
                        'scale': [0, 1]
                    }
                }
            ]
        },
        'training_settings': {
            'learning_rate': 0.05,
            'max_epochs': 200,
            'momentum': 0.1,
            'early_stopping': {
                'min_accuracy': 97,
                'best_weights_delay': {
                    'delay': 30
                },
                'raised_error': {
                    'threshold': 0.65,
                    'n_previous_epochs': 30
                },
                'accumulated_error': {
                    'threshold': 0.003,
                    'n_previous_epochs': 30
                }
            }
        }
    },
    {
        'id': 3,
        'architecture': {
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
                    'function': 'tanh',
                    'weights': {
                        'distribution': 'uniform',
                        'scale': [0, 1]
                    }
                },
            ]
        },
        'training_settings': {
            'learning_rate': 0.05,
            'max_epochs': 200,
            'momentum': 0.1,
            'early_stopping': {
                'min_accuracy': 97,
                'best_weights_delay': {
                    'delay': 30
                },
                'raised_error': {
                    'threshold': 0.65,
                    'n_previous_epochs': 30
                },
                'accumulated_error': {
                    'threshold': 0.003,
                    'n_previous_epochs': 30
                }
            }
        }
    },
    {
        'id': 4,
        'architecture': {
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
                        'distribution': 'uniform',
                        'scale': [0, 1]
                    }
                },
                {
                    'neurons': 15,
                    'function': 'tanh',
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
            ]
        },
        'training_settings': {
            'learning_rate': 0.05,
            'max_epochs': 200,
            'momentum': 0.1,
            'early_stopping': {
                'min_accuracy': 97,
                'best_weights_delay': {
                    'delay': 30
                },
                'raised_error': {
                    'threshold': 0.65,
                    'n_previous_epochs': 30
                },
                'accumulated_error': {
                    'threshold': 0.003,
                    'n_previous_epochs': 30
                }
            }
        }
    }
]