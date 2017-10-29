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
    estimate_inputs = None
    estimate_labels = None
    test_inputs = None
    test_labels = None
    dims = 0

    def __init__(self):
        pass

    def load_data(self, train_file, test_file, train_split):
        self.load_train_data(train_file, train_split)
        self.load_test_data(test_file)

    def load_train_data(self, file, split):
        inputs, labels, count = self.__load_data(file)

        # split to train and estimate set
        indices = np.arange(count)
        random.shuffle(indices)
        split = int(count / 100 * split)
        train_indices = indices[:split]
        estimate_indices = indices[split:]

        self.train_inputs = inputs[:, train_indices]
        self.train_labels = labels[train_indices]
        self.estimate_inputs = inputs[:, estimate_indices]
        self.estimate_labels = labels[estimate_indices]

    def load_test_data(self, file):
        self.test_inputs, self.test_labels, _ = self.__load_data(file)

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