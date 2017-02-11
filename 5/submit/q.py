import sys
import time
import operator
import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw5 import *
from scipy.misc import logsumexp
from numpy.linalg import norm

stop_crit_threshold = 0.0001

def norm_square(x):
    n = norm(x)
    return n * n

#Perform an EM step
#ss stands for sigma squared
#c stands for the prior
#Modifies mu, ss, c vectors
def do_em_step____log(x, mu, ss, c):
    #Get n,k consts from input
    n = len(x)
    k = len(mu)

    #Calc p(z_i = m | x_i, theta)
    p = np.zeros((n, k))
    for i in range(n):
        for m in range(k):
            p[i, m] = log((2*pi)) * (-k/2.0)
            p[i, m] += log(ss[m]) * (-1.0/2.0)
            p[i, m] += (-(norm_square(x[i]-mu[m]))/(2*ss[m]))
        p[i,:] -= logsumexp(p[i])
    ep = e**p

    #Calc new c
    for m in range(k):
        c[m] = (1.0/n) * ep[:,m].sum()

    #Calc new mu
    for m in range(k):
        mu[m] = 0
        tmp = np.zeros((n, len(x[0])))
        for i in range(n):
            tmp[i] = p[i, m] + log(x[i])
        mu[m] = e ** (logsumexp(tmp, axis=0) - logsumexp(p[:,m]))

    #Calc new ss
    for m in range(k):
        ss[m] = 0
        tmp = np.zeros(n)
        for i in range(n):
            tmp[i] = p[i, m] + log(norm_square(x[i] - mu[m]))
        ss[m] = e ** (logsumexp(tmp) - logsumexp(p[:,m]))

def do_em_step(x, mu, ss, c):
    #Get n,k consts from input
    n = len(x)
    k = len(mu)

    #Calc p(z_i = m | x_i, theta)
    p = np.zeros((n, k))
    for i in range(n):
        for m in range(k):
            p[i, m] = log((2*pi)**(-k/2.0))
            p[i, m] += log(ss[m] ** (-1.0/2.0))
            p[i, m] += (-(norm(x[i]-mu[m]))/(2*ss[m]))
        p[i,:] -= logsumexp(p[i])
    p = e**p

    #Calc new c
    for m in range(k):
        c[m] = (1.0/n) * p[:,m].sum()

    #Calc new mu
    for m in range(k):
        mu[m] = 0
        for i in range(n):
            mu[m] += p[i, m] * x[i]
        mu[m] /= p[:,m].sum()

    #Calc new ss
    for m in range(k):
        ss[m] = 0
        for i in range(n):
            ss[m] += p[i, m] * norm(x[i] - mu[m])
        ss[m] /= p[:,m].sum()

#Classify according to clusters
def classify(mu, ss, c, x):
    k = len(c)
    max_prob = -1e99
    best_cluster = -1

    for i in range(k):
        #prob = ss[i]**(-1.0/2.0) * e**(-(norm_square(x-mu[i]))/(2*ss[i]))
        prob = log(ss[i])*(-1.0/2.0) + (-(norm_square(x-mu[i]))/(2*ss[i]))
        if False:
            print "norm_square(x-mu[i])", norm_square(x-mu[i])
            print "ss[i]", ss[i]
            print "prob:", prob
        if prob > max_prob:
            max_prob = prob
            best_cluster = i

    return best_cluster

#Measure accuracy on the test set
def measure_accuracy(mu, ss, c, labels):
    errors = 0
    for i in range(len(test_data)):
        if labels[classify(mu, ss, c, test_data[i])] != test_labels[i]:
            errors += 1
    return 1 - float(errors) / float(len(test_data))

#Calculate the likelihood
def calc_likelihood(x, mu, ss, c):
    k = len(c)
    log_likelihoods = []
    for i in range(len(x)):
        single_point_log_likelihoods = []
        for m in range(len(c)):
            l = log((2*pi)) * (-k/2.0)
            l += log(ss[m]) * (-1.0/2.0)
            l += (-(norm_square(x[i]-mu[m]))/(2*ss[m]))
            single_point_log_likelihoods.append(l)
        log_likelihoods.append(logsumexp(single_point_log_likelihoods))
    return np.sum(log_likelihoods)

#Use the implementation above to produce plots and measurements for question 4
def answer(filenames):
    #Init parameters
    k = 5
    t = 0
    x = train_data
    c = np.ones(k) / float(k)
    ss = np.ones(k) * train_data.var(axis=1).mean()
    mu = np.random.randint(0, 256, (k, len(train_data[0]))).astype(train_data.dtype)

    if True:
        print "******************************"
        print "lets take the best sigmas and vars"
        keys = [0, 1, 3, 4, 8]
        for i in range(k):
            m = keys[i]
            c[i] = float(len(test_labels == m)) / float(len(test_labels))
            mu[i] = test_data[test_labels == m].mean(axis=0)
            ss[i] = test_data[test_labels == m].var(axis=1).mean()
            ss[i] = 10

        print "what is the likelihood of this perfection?"
        print "its", calc_likelihood(x, mu, ss, c)

        print "and the accuracy is", measure_accuracy(mu, ss, c, keys)
        print "******************************"

    #Save likelihood for plot
    likelihood = []

    while True:
        t += 1
        start = time.time()
        old_mu = mu.copy()
        do_em_step(x, mu, ss, c)
        stop_crit = norm(mu-old_mu, axis=1).sum()
        likelihood.append(calc_likelihood(x, mu, ss, c))
        elapsed = time.time() - start
        print "itreation", t, "elapsed:", elapsed, "stop_crit:", stop_crit, "likelihood:", likelihood[-1]
        if stop_crit < stop_crit_threshold or math.isnan(stop_crit):
            print "reached stop criterion"
            break

    #label clusters for accuracy measurement
    #each cluster of train_data points is labeled as its most common train_label
    #we are using the train_labels, and afterwards test on the test_data + test_labels

    votes = {}
    for i in range(k):
        votes[i] = {0:0}

    for i in range(len(train_data)):
        cluster_index = classify(mu, ss, c, train_data[i])
        cluster_label = train_labels[i]
        votes[cluster_index][cluster_label] = votes[cluster_index].get(cluster_label, 0) + 1

    #print votes

    labels = []
    for i in range(k):
        most_common_label_num_votes = max(votes[i].values())
        for label in votes[i].keys():
            if votes[i][label] == most_common_label_num_votes:
                most_common_label = label
        labels.append(most_common_label)

    print "Accuracy:", measure_accuracy(mu, ss, c, labels)

    #Plot likelihood
    plt.plot(range(1,t+1), likelihood, 'ko')
    plt.savefig(filenames[0])

    #Plot clusters
    for m in range(k):
        print "cluster number", m, "c:", c[m], "ss:", ss[m]
        plt.imshow(mu[m].reshape(28,28), interpolation='nearest')
        plt.savefig(filenames[1+m])

if True:
    if len(sys.argv) > 1 and sys.argv[1] == '4':
        #We output 6 plots for subquestions b-e
        #Plot the likelihood, plot 5 clusters
        filenames = [None] * 6
        filenames[0] = sys.argv[2]
        filenames[1] = sys.argv[3]
        filenames[2] = sys.argv[4]
        filenames[3] = sys.argv[5]
        filenames[4] = sys.argv[6]
        filenames[5] = sys.argv[7]
        answer(filenames)
    else:
        print "Error: please choose a valid command"

#TODO: fix likelihood, restore iterations, remove fake init
