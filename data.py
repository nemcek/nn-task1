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
    estimation_inputs = None
    estimation_labels = None
    validation_inputs = None
    validation_labels = None
    test_inputs = None
    test_labels = None
    dims = 0

    def __init__(self):
        pass

    def load_data(self, train_file, test_file, train_split, normalize=True):
        self.load_train_data(train_file, train_split, normalize)
        self.load_test_data(test_file, normalize)

    def load_train_data(self, file, split, normalize):
        inputs, labels, count = self.__load_data(file)

        # split to train and estimate set
        indices = np.arange(count)
        random.shuffle(indices)
        split = int(count / 100 * split)
        train_indices = indices[:split]
        estimate_indices = indices[split:]
        
        if normalize:
            for i in range(self.dims):
                inputs[i] = DataOperations.normalize(inputs[i])

        self.estimation_inputs = inputs[:, train_indices]
        self.estimation_labels = labels[train_indices]
        self.validation_inputs = inputs[:, estimate_indices]
        self.validation_labels = labels[estimate_indices]

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

    def normalize_data(self):
        # TODO: Normalize train data as whole and after that split them
        self.estimation_inputs[0] = DataOperations.normalize(self.estimation_inputs[0])
        self.estimation_inputs[1] = DataOperations.normalize(self.estimation_inputs[1])
        self.validation_inputs[0] = DataOperations.normalize(self.estimate_inputs[0])
        self.validation_inputs[1] = DataOperations.normalize(self.estimate_inputs[1])
        self.test_inputs[0] = DataOperations.normalize(self.test_inputs[0])
        self.test_inputs[0] = DataOperations.normalize(self.test_inputs[1])


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