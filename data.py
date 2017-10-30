"""
    Data handling and manipulations
"""

#!/usr/bin/env python
import random
import numpy as np

"""
    Class handling input data, spliting into train and estimation sets
"""
class Data:
    train_inputs = None 
    train_labels = None
    test_inputs = None 
    test_labels = None 
    dims = 0
    train_count = 0

    def __init__(self):
        pass

    def load_data(self, train_file, test_file, train_splits, normalize=True):
        self.load_train_data(train_file, train_splits, normalize)
        self.load_test_data(test_file, normalize)

    def load_train_data(self, file, splits, normalize):
        self.train_inputs, self.train_labels, self.train_count = self.__load_data(file)

        if normalize:
            for i in range(self.dims):
                self.train_inputs[i] = DataOperations.normalize(self.train_inputs[i])

    def load_test_data(self, file, normalize):
        self.test_inputs, self.test_labels, _ = self.__load_data(file)

        if normalize:
            for i in range(self.dims):
                self.test_inputs[i] = DataOperations.normalize(self.test_inputs[i])

    def __load_data(self, file):
        data = np.loadtxt(file, dtype=str, skiprows=1).T
        inputs = data[:-1]
        labels = data[-1]
        (dim, count) = data.shape
        self.dims = dim - 1

        # cast all inputs and labels to float / int
        inputs = np.array(list(map(lambda row: np.array(list(map(float, row))), inputs)))
        labels = np.array(list(map(ord, labels)))

        return inputs, labels, count

    def split(self, validation_index):
        estimation_inputs = []
        estimation_labels = []
        validation_inputs = []
        validation_labels = []

        for i in range(len(self.train_inputs_set)):
            if i == validation_index:
                validation_inputs = self.train_inputs_set[i]
                validation_labels = self.train_labels_set[i]
            else:
                if len(estimation_inputs) == 0:
                    estimation_inputs = self.train_inputs_set[i]
                    estimation_labels = self.train_labels_set[i]
                else:
                    estimation_inputs = np.concatenate((estimation_inputs, self.train_inputs_set[i]))
                    estimation_labels = np.concatenate((estimation_labels, self.train_labels_set[i]))
        
        return estimation_inputs, estimation_labels, validation_inputs, validation_labels

class DataOperations:

    _DISTRIB_FUNCTIONS = {
        'uniform': np.random.rand,
        'gauss': np.random.randn,
    }

    @staticmethod
    def normalize(data):
        return (data - np.mean(data)) / np.std(data)

    @staticmethod
    def scale(data, minVal, maxVal):
        return data * (maxVal - minVal) + minVal

    @staticmethod
    def generate_distribution(dist, dims, scale):
        return(DataOperations.scale(DataOperations._DISTRIB_FUNCTIONS[dist](dims[0], dims[1]), scale[0], scale[1]))