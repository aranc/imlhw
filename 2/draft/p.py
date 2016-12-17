import sys
import time
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw2 import *

def build_preceptron(samples, labels):
    w = np.zeros(len(samples[0]))
    for i in range(len(samples)):
        predicted = classify(w, samples[i])
        if predicted != labels[i]:
            w += labels[i] * samples[i] / np.linalg.norm(samples[i])
    return w

def classify(w, x):
    x = x / np.linalg.norm(x)
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

def q1a_helper(train, train_labels, test, test_labels):
    w = build_preceptron(train, train_labels)
    errors = 0
    for i in range(len(test)):
        predicted = classify(w, test[i])
        if predicted != test_labels[i]:
            errors += 1
    return 1 - float(errors)/float(len(test_data))

def q1a(n=5, times=100):
    acc = np.zeros(times)
    for i in range(times):
        idx = np.random.permutation(n)
        acc[i] = q1a_helper(train_data[idx], train_labels[idx], test_data, test_labels)
    return np.mean(acc)

def q1a_all():
    for n in (5, 10, 50, 100, 500, 1000, 5000):
        print n, q1a(n)

def q1b():
    w = build_preceptron(train_data, train_labels)
    plt.imshow(w.reshape(28,28))
    plt.show()
    return w

def q1c():
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
            print test_labels[i]
            plt.imshow(test_data[i].reshape(28,28))
            plt.show()
    print "done. success rate:", 1 - float(errors)/float(len(test_data)), "elapsed:", time.time() - start

import sklearn.svm
def q2():
    S=sklearn.svm.LinearSVC(loss='hinge', fit_intercept=False, C=1.0)
    S.fit(train_data, train_labels)
    errors = 0
    for i in range(len(test_data)):
        predicted = S.predict(test_data[i].reshape(1, -1))
        if predicted != test_labels[i]:
            errors += 1
    print "done. success rate:", 1 - float(errors)/float(len(test_data))
    return S.coef_

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
