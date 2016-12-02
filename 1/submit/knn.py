from __future__ import division
import numpy as np
import time

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

import numpy.random
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

#Return the most common digit in array
def mode(array):
    count = np.zeros(10)
    for i in range(10):
        count[i] = np.count_nonzero(array == i)
    return np.argmax(count)

#Implement KNN
def knn(images, labels, image, k):
    dists = np.linalg.norm(images - image, axis=1)
    nearest = np.argsort(dists)[:k]
    return mode(labels[nearest])

#Measure KNN
def measure_knn(k=10, n=1000, verbose=True):
    start = time.time()
    bad = 0
    for i in range(len(test)):
        image = test[i]
        label = test_labels[i]
        predicted = knn(train[:n], train_labels[:n], image, k)
        if predicted != label:
            bad += 1
    correct_ratio = 1 - float(bad)/float(len(test))
    end = time.time()
    if verbose:
        print "k:", k, "n:", n, "correct_ratio:", correct_ratio, "elapsed:", end - start
    return correct_ratio
