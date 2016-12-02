from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import pickle

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

import numpy.random
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :]
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :]
test_labels = labels[idx[10000:]]

def go():
    for i in range(1000):
        for j in range(1000):
            if np.array_equal(train[i,:],test[j,:]):
                print "train",i,"== test",j

go()
