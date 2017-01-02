import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw3 import *

#Classify for question 6
def svm_sgd_classify(w, x):
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
    indicator_neg = np.asmatrix(np.ones((k,k)) - np.eye(k))
    #Column p has 0s in all places except p
    indicator_pos = np.asmatrix(np.eye(k))

    #Init a new weights matrix
    w = np.asmatrix(np.zeros((k, d)))

    for iteration in range(T):
        #Sample a random point
        i = np.random.randint(0, m)
        xi = np.asmatrix(train_data[i]).T
        yi = int(train_labels[i])
        #Find argmax(w_p*x - w_yi*x + 1(p!=yi))
        j_max = np.argmax(w*xi - w[yi]*xi + indicator_neg[:,yi])
        #Update weights
        w *= (1 - eta)
        update_vector = np.multiply(indicator_pos[:,yi], indicator_neg[:,j_max])
        update_vector -= np.multiply(indicator_neg[:,yi], indicator_pos[:,j_max])
        w += update_vector * xi.T * C * eta

    #Output weights matrix
    return w


#Classify for question 7
def svm_kernel_classify(K, train_data, M, x):
    #Return argmax(\sum{i=1}{m} M_ji*K(xi, x))
    kernel_vector = np.fromfunction(np.vectorize(lambda i: K(train_data[int(i)], x)), (train_data.shape[0],))
    #Use matrix notation
    kernel_vector = np.asmatrix(kernel_vector).T
    #Return argmax as specified
    return np.argmax(M*kernel_vector)

#Train for question 7
def svm_kernel_train(K, train_data, train_labels, T, eta, C):
    m = train_data.shape[0]
    #Constant number of labels
    k=10

    #kernel_matrix = np.fromfunction(np.vectorize(lambda i, j: K(train_data[int(i)], train_data[int(j)])), (m, m))
    #TODO: dont precalculate, too limiting

    #Init M to zeroes
    M = np.asmatrix(np.zeros((k, m)))

    for iteration in range(T):
        #Sample a random point
        i = np.random.randint(0, m)
        xi = train_data[i]
        yi = int(train_labels[i])
        #Find argmax(\sum{t=1}{m} M_jt*K(xt, xi))
        kernel_vector = np.fromfunction(np.vectorize(lambda t: K(train_data[int(t)], train_data[int(i)])), (m,))
        j_max = argmax(M*np.asmatrix(kernel_vector).T)
        #update M
        M *= (1 - eta)
        for j in range(k):
            if j != yi and j == j_max:
                M[j,i] -= eta*C
            if j == yi and j != j_max:
                M[j,i] += eta*C

    #Output M matrix
    return M
        
def go1():
    w=svm_sgd_train(train_data, train_labels, 10000, 10**-4, .001)
    print "done building"
    errors = 0
    for i in range(len(test_data)):
        predicted = svm_sgd_classify(w, test_data[i])
        if predicted != int(test_labels[i]):
            errors += 1
    print "1:", 1 - float(errors)/float(len(test_data))
def go2():
    K = lambda x1, x2: np.dot(x1,x2)
    start = time.time()
    M=svm_kernel_train(K, train_data, train_labels, 10000, 10**-4, .001)
    print "done building:", time.time() - start
    errors = 0
    start = time.time()
    for i in range(len(test_data)):
        predicted = svm_kernel_classify(K, train_data, M, test_data[i])
        if predicted != int(test_labels[i]):
            errors += 1
    print "done testing:", time.time() - start
    print "2:", 1 - float(errors)/float(len(test_data))
go1()
go2()
