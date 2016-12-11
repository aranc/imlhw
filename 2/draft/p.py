import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw2 import *

def build_preceptron(samples, labels):
    w = np.zeros(len(samples[0]))
    for i in range(len(samples)):
        predicted = classify(w, samples[i])
        if predicted != labels[i]:
            w += labels[i] * samples[i]

def classify(w, x):
    return 1 if np.dot(w, x) >= 0 else -1

def go():
    print "building preceptron"
    start = time.time()
    w = build_preceptron(train_data, train_labels)
    print "done. elapsed:", time.time() - start

    print "testing"
    start = time.time()
    errors = 0
    for i in range(len(test_data)):
        predicted = classify(w, test_data[i])
        if predicted != test_labels[i]:
            errors += 1
    print "done. success rate:", 1 - float(errors)/float(len(test_data)), "elapsed:", time.time() - start


