import sys
import time
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw2 import *

def build_sgd(train_data, train_labels, C, mu0, T):
    d = len(train_data[0])
    m = len(train_data)
    w = np.zeros(d)

    for t in range(1, T + 1):
        mut = float(mu0) / float(t)
        i = np.random.randint(0, m)
        xi = train_data[i]
        xi = xi / np.linalg.norm(xi)
        yi = train_labels[i]
        if yi * np.dot(w, xi) < 1:
            w = (1 - mut) * w + mut * C * yi * xi

    return w

def classify(w, x):
    x = x / np.linalg.norm(x)
    return 1 if np.dot(w, x) >= 0 else -1

def measure_sgd(w, data, labels):
    errors = 0
    for i in range(len(data)):
        predicted = classify(w, data[i])
        if predicted != labels[i]:
            errors += 1
    return 1 - float(errors)/float(len(data))

def go():
    print "building SGD"
    start = time.time()
    w = build_sgd(train_data, train_labels, C=1, mu0=1, T=1000)
    print "elapsed:", time.time() - start

    print "testing"
    start = time.time()
    acc = measure_sgd(w, test_data, test_labels)
    print "acc:", acc, "elapsed:", time.time() - start

go()
