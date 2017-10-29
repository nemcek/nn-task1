"""
proj.py
"""

#!/usr/bin/env python
from data import *
from classifier import *

data = Data()
data.load_data('data/train.dat', 'data/test.dat', 20)

model = MLPClassifier(data.dims, 20, np.max(data.train_labels)+1)
trainCEs, trainREs = model.train(data.train_inputs, data.train_labels, alpha=0.05, eps=500, trace=True, trace_interval=10)
 
testCE, testRE = model.test(data.test_inputs, data.test_labels)
print('Final testing error: CE = {:6.2%}, RE = {:.5f}'.format(testCE, testRE))

plot_both_errors(trainCEs, trainREs, testCE, testRE, block=False) 