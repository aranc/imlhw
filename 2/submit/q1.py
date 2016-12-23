import sys
import time
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw2 import *

#Build perceptron from training set
def build_perceptron(samples, labels):
    w = np.zeros(len(samples[0]))
    for i in range(len(samples)):
        predicted = classify(w, samples[i])
        if predicted != labels[i]:
            w += labels[i] * samples[i] / np.linalg.norm(samples[i])
    return w

#Classify sample x using weights w
def classify(w, x):
    x = x / np.linalg.norm(x)
    return 1 if np.dot(w, x) >= 0 else -1

#Test / Validate weights w
def measure(w, data, labels):
    errors = 0
    for i in range(len(data)):
        predicted = classify(w, data[i])
        if predicted != labels[i]:
            errors += 1
    return 1 - float(errors)/float(len(data))


#Run a measurement for question 1a, for a specific n value
def q1a_helper(n):
    T=100
    data = train_data[:n]
    labels = train_labels[:n]
    res = np.zeros(T)
    for i in range(T):
        idx = np.random.permutation(n)
        w = build_perceptron(data[idx], labels[idx])
        res[i] = measure(w, test_data, test_labels)
    res = np.sort(res)
    #return mean, 5 precentile, and 95 precentile
    return np.mean(res), res[5], res[95]

#Print table for question 1a
def q1a():
    print "List of n, mean accuracy, 5 precentile, and 95 precentile"
    for n in [5, 10, 50, 100, 500, 1000, 5000]:
        mean, p5, p95 = q1a_helper(n)
        print n, mean, p5, p95

def _q1b(output=None):
    w=build_perceptron(train_data, train_labels)
    plt.imshow(w.reshape(28,28))
    if output == None:
        plt.show()
    else:
        plt.savefig(output)

def q1b():
    w=build_perceptron(train_data, train_labels)
    plt.imshow(w.reshape(28,28))
    plt.show()

if True:
    #Get subquestion from first argument
    if sys.argv[1] == 'a':
        q1a()
    elif sys.argv[1] == 'b':
        q1b()
    if sys.argv[1] == 'c':
        output = sys.argv[2]
        q3c(output)
    if sys.argv[1] == 'd':
        print q3d()
    else:
        print "Error: please choose subquestion (a,b,c,d)"
