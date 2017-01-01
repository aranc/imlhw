import sys
import time
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw3 import *

#Classify for question 6
def svm_sgd_classify(w, x):
    #Assert shape
    k = w.shape[0]
    assert k == 10, "expecting w.shape[0]==10"
    #Use matrix notation
    x = np.asmatrix(x).T
    #Return argmax(w_j * x)
    return np.argmax(w*x)

#Train for question 6
def svm_sgd_train(train_data, train_labels, T, C, eta):
    m = train_data.shape[0]
    d = train_data.shape[1]
    k = 10

    #Column p has 1s in all places except p
    indicator = np.ones((k,k)) - np.eye(k)

    #Init a new weights matrix
    w = np.asmatrix(np.zeroes((k, d)))

    for iteration in range(T):
        #Sample a random point
        i = np.random.randint(0, m)
        xi = np.asmatrix(train_data[i]).T
        yi = train_label[i]
        #w = (1-eta)w
        w *= (1 - eta)
        #Find argmax(w_p*x - w_yi*x + 1(p!=yi))
        j_max = np.argmax(w*xi - w[yi]*xi + indicator(yi))

    #Output weights matrix
    return w


#Classify for question 7
def svm_kernel_classify():
    pass

#Train for question 7
def svm_kernel_train():
    pass
