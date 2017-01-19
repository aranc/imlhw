import sys
import time
import operator
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import svd
from hw4 import *

#Shared code that answers subquestions a,b,c
def subquestions_abc(filnames, data):
    #Plot the mean
    mean = np.mean(data, 0)
    plt.imshow(mean.reshape(28,28), interpolation='nearest')
    plt.savefig(filenames[0])

    #Call SVD
    u, s, v = svd(data - mean)

    #Plot 5 first eigenvectors
    for i in range(5):
        plt.imshow(v[i].reshape(28,28), interpolation='nearest')
        plt.savefig(filenames[1+i])

#Answers subquestion a
def subquestion_a(filenames):
    subquestions_abc(filenames, train_data[train_labels == pos])

#Answers subquestion b
def subquestion_b(filenames):
    subquestions_abc(filenames, train_data[train_labels == neg])

#Answers subquestion c
def subquestion_c(filenames):
    subquestions_abc(filenames, train_data)

if True:
    if len(sys.argv) > 1 and sys.argv[1] == 'a':
        #We output 7 plots for subquestions a,b, and c.
        #Plot the mea, plot 5 eigenvectors, and plot the eigenvalues
        filenames = [None] * 7
        filenames[0] = sys.argv[2]
        filenames[1] = sys.argv[3]
        filenames[2] = sys.argv[4]
        filenames[3] = sys.argv[5]
        filenames[4] = sys.argv[6]
        filenames[5] = sys.argv[7]
        filenames[6] = sys.argv[8]
        subquestion_a(filenames)
    elif len(sys.argv) > 1 and sys.argv[1] == 'b':
        #We output 7 plots for subquestions a,b, and c.
        #Plot the mea, plot 5 eigenvectors, and plot the eigenvalues
        filenames = [None] * 7
        filenames[0] = sys.argv[2]
        filenames[1] = sys.argv[3]
        filenames[2] = sys.argv[4]
        filenames[3] = sys.argv[5]
        filenames[4] = sys.argv[6]
        filenames[5] = sys.argv[7]
        filenames[6] = sys.argv[8]
        subquestion_c(filenames)
    elif len(sys.argv) > 1 and sys.argv[1] == 'c':
        #We output 7 plots for subquestions a,b, and c.
        #Plot the mea, plot 5 eigenvectors, and plot the eigenvalues
        filenames = [None] * 7
        filenames[0] = sys.argv[2]
        filenames[1] = sys.argv[3]
        filenames[2] = sys.argv[4]
        filenames[3] = sys.argv[5]
        filenames[4] = sys.argv[6]
        filenames[5] = sys.argv[7]
        filenames[6] = sys.argv[8]
        subquestion_c(filenames)
    else:
        print "Error: please choose a valid command"
