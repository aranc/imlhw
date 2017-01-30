import sys
import time
import operator
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw5 import *
from scipy.misc import logsumexp

#####TODO#######
#test for subquestion e
#calc likelihood for another subquestion

#Perform an EM step
#ss stands for sigma squared
#c stands for the prior
#Modifies mu, ss, c vectors
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
            p[i, m] += (-(np.norm(x[i]-mu[m]))/(2*ss[m]))
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
            ss[m] += p[i, m] * np.norm(x[i] - mu[m]
        ss[m] /= p[:m].sum()

#Use the implementation above to produce plots and measurements for question 4
def answer(filenames):
    #Init parameters
    k = 5
    t = 0
    x = train_data
    c = np.ones(k) / float(k)
    ss = np.ones(k) * train_data.var(axis=1).mean()
    mu = np.random.randint(0, 256, (k, len(train_data[0])))

    #Save likelihood for plot
    likelihood = []

    while True:
        old_mu = mu.copy()
        do_em_step(x, mu, ss, c)
        stop_crit = (mu-old_mu).mean(axis=1).sum()
        likelihood.append(calc_likelihood(x, mu, ss, c))
        print "itreation", t, "stop_crit:", stop_crit, "likelihood:", likelihood[-1]
        if stop_crit < 1:
            print "reached stop criterion"
            break 

    #Plot likelihood
        plt.plot(range(1,t+1), likelihood, 'ko')
        plt.savefig(filenames[1+m])

    #Plot clusters
    for m in range(k):
        print "cluster number", m, "c:", c[m], "ss:", ss[m]
        plt.imshow(mu[m].reshape(28,28), interpolation='nearest')
        plt.savefig(filenames[1+m])

if False:
    if len(sys.argv) > 1 and sys.argv[1] == '4':
        #We output 6 plots for subquestions b-e
        #Plot the likelihood, plot 5 clusters
        filenames = [None] * 4
        filenames[0] = sys.argv[2]
        filenames[1] = sys.argv[3]
        filenames[2] = sys.argv[4]
        filenames[3] = sys.argv[5]
        filenames[4] = sys.argv[6]
        filenames[5] = sys.argv[7]
        answer(filenames)
    else:
        print "Error: please choose a valid command"
