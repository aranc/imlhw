import sys
import time
import operator
import random
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

#Answers subquestion d
def subquestion_d(filename):
    #Call SVD
    u, s, v = svd(train_data)

    red_points = []
    blue_points = []

    #Project images on first and second eigenvctors
    for i in range(len(train_data)):
        x = np.dot(v[0], train_data[i])
        y = np.dot(v[1], train_data[i])
        if train_labels[i] == 1:
            red_points.append((x,y))
        elif train_labels[i] == -1:
            blue_points.append((x,y))

    #Plot colored points
    plt.plot([p[0] for p in red_points], [p[1] for p in red_points], '.r') 
    plt.plot([p[0] for p in blue_points], [p[1] for p in blue_points], '.b') 

    plt.savefig(filename)

#Answers subquestion e
def subquestion_e(filename):
    #Call SVD
    u, s, v = svd(train_data)

    #Choose a random image
    i = random.randint(0, len(train_data) - 1)

    #Calc image projection on first k principal axes
    def project_image(k):
        p = np.matrix(v[:k])
        return p*p.T*train_data[i]

    #Plot images
    ax = plt.subplot("141")
    ax.set_title("Original image")
    ax.imshow(train_data[i].reshape((28,28)))
    ax = plt.subplot("142")
    ax.set_title("k=10")
    ax.imshow(project_image(10).reshape((28,28)))
    ax = plt.subplot("143")
    ax.set_title("k=30")
    ax.imshow(project_image(30).reshape((28,28)))
    ax = plt.subplot("144")
    ax.set_title("k=50")
    ax.imshow(project_image(50).reshape((28,28)))
    
    plt.show()
    #plt.savefig(filename)

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
    elif len(sys.argv) > 1 and sys.argv[1] == 'd':
        filename = sys.argv[2]
        subquestion_d(filename)
    elif len(sys.argv) > 1 and sys.argv[1] == 'e':
        filename = sys.argv[2]
        subquestion_e(filename)
    elif len(sys.argv) > 1 and sys.argv[1] == 'debug':
        pass
    else:
        print "Error: please choose a valid command"
