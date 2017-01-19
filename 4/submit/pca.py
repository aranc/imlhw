import sys
import time
import operator
import numpy as np
import matplotlib
#matplotlib.use('Agg')
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
        plt.clf()
        plt.imshow(v[i].reshape(28,28), interpolation='nearest')
        plt.savefig(filenames[1+i])

    #Plot eigenvalues
    plt.clf()
    plt.plot(range(1,101), s[:100], '*-')
    plt.savefig(filenames[6])


#Answers subquestion a
def subquestion_a(filenames):
    subquestions_abc(filenames, train_data[train_labels == 1])

#Answers subquestion b
def subquestion_b(filenames):
    subquestions_abc(filenames, train_data[train_labels == -1])

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
        subquestion_b(filenames)
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
    elif len(sys.argv) > 1 and sys.argv[1] == 'debug':
        data1 = train_data[train_labels==-1][0]
        data2 = train_data[train_labels==-1][1]
        mean = np.mean(np.vstack((data1, data2)), 0)
        plt.clf()
        plt.imshow(data1.reshape(28,28), interpolation='nearest')
        plt.show()
        plt.clf()
        plt.imshow(data2.reshape(28,28), interpolation='nearest')
        plt.show()
        plt.clf()
        plt.imshow(mean.reshape(28,28), interpolation='nearest')
        plt.show()
    else:
        print "Error: please choose a valid command"
