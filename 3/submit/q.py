import sys
import time
import operator
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
def svm_kernel_train(K, train_data, train_labels, T, C, eta):
    m = train_data.shape[0]
    #Constant number of labels
    k=10

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

#Take a classifier for question 6, and measure it on a given set
def svm_sgd_measure(w, data, labels):
    errors = 0
    for i in range(len(data)):
        predicted = svm_sgd_classify(w, data[i])
        if predicted != int(labels[i]):
            errors += 1
    return 1 - float(errors)/float(len(data))

#Take a classifier for question 7, and measure it on a given set
def svm_kernel_measure(K, M, xi, data, labels):
    errors = 0
    for i in range(len(data)):
        predicted = svm_kernel_classify(K, xi, M, data[i])
        if predicted != int(labels[i]):
            errors += 1
    return 1 - float(errors)/float(len(data))

#Utility function, prints max 10 elements in a hash
def print_best_10(_dict):
    print "max 10 values:"
    sorted_dict = sorted(_dict.items(), key=operator.itemgetter(1))
    for k, v in sorted_dict[-10:]:
        print k,v

#Plot the training and the validation errors for various values of eta
def svm_sgd_find_eta(_from, _to, _step, C, T, output=None):
    values_range = np.arange(_from, _to, _step)
    training_accuracy = {}
    validation_accuracy = {}
    for value in values_range:
        eta = 10**value
        w = svm_sgd_train(train_data, train_labels, T, C, eta)
        training_accuracy[value] = svm_sgd_measure(w, train_data, train_labels)
        validation_accuracy[value] = svm_sgd_measure(w, validation_data, validation_labels)
        print "value:", value, "training accuracy:", training_accuracy[value], "validation accuracy:", validation_accuracy[value]
    plt.title("Find eta. Use C="+str(C)+" and T="+str(T))
    plt.gca().set_xlabel("log10(eta)")
    plt.gca().set_ylabel("Accuracy")
    plt.plot(values_range, [validation_accuracy[value] for value in values_range], 'ko', label="validation")
    plt.plot(values_range, [training_accuracy[value] for value in values_range], 'k*', label="training")
    plt.legend()
    if output == None:
        plt.show()
    else:
        plt.savefig(output)
    print_best_10(validation_accuracy)

#Plot the training and the validation errors for various values of eta
def svm_sgd_find_C(_from, _to, _step, eta, T, output=None):
    values_range = np.arange(_from, _to, _step)
    training_accuracy = {}
    validation_accuracy = {}
    for value in values_range:
        C = 10**value
        w = svm_sgd_train(train_data, train_labels, T, C, eta)
        training_accuracy[value] = svm_sgd_measure(w, train_data, train_labels)
        validation_accuracy[value] = svm_sgd_measure(w, validation_data, validation_labels)
        print "value:", value, "training accuracy:", training_accuracy[value], "validation accuracy:", validation_accuracy[value]
    plt.title("Find C. Use eta="+str(eta)+" and T="+str(T))
    plt.gca().set_xlabel("log10(C)")
    plt.gca().set_ylabel("Accuracy")
    plt.plot(values_range, [validation_accuracy[value] for value in values_range], 'ko', label="validation")
    plt.plot(values_range, [training_accuracy[value] for value in values_range], 'k*', label="training")
    plt.legend()
    if output == None:
        plt.show()
    else:
        plt.savefig(output)
    print_best_10(validation_accuracy)

#Plot weights for a specific digit
def svm_sgd_show_digit(C, eta, T, digit, output=None):
    w = svm_sgd_train(train_data, train_labels, T, C, eta)
    plt.imshow(w[digit].reshape(28,28), interpolation='nearest')
    plt.title(str(digit))
    if output == None:
        plt.show()
    else:
        plt.savefig(output)

#Calc accuracy for question 6 on the test data
def svm_sgd_calc_accuracy(C, eta, T):
    w = svm_sgd_train(train_data, train_labels, T, C, eta)
    return svm_sgd_measure(w, validation_data, validation_labels)
    return svm_sgd_measure(w, test_data, test_labels)

if True:
    #Get subquestion from first argument
    if sys.argv[1] == '6':
        if sys.argv[2] == 'find_eta':
            _from = float(sys.argv[3])
            _to = float(sys.argv[4])
            _step = float(sys.argv[5])
            C = 10**float(sys.argv[6])
            T = int(sys.argv[7])
            filename = sys.argv[8]
            svm_sgd_find_eta(_from, _to, _step, C, T, filename)
        elif sys.argv[2] == 'find_C':
            _from = float(sys.argv[3])
            _to = float(sys.argv[4])
            _step = float(sys.argv[5])
            eta = 10**float(sys.argv[6])
            T = int(sys.argv[7])
            filename = sys.argv[8]
            svm_sgd_find_C(_from, _to, _step, eta, T, filename)
        elif sys.argv[2] == 'show_digit':
            C = 10**float(sys.argv[3])
            eta = 10**float(sys.argv[4])
            T = int(sys.argv[5])
            digit = int(sys.argv[6])
            filename = sys.argv[7]
            svm_sgd_show_digit(C, eta, T, digit, filename)
        elif sys.argv[2] == 'calc_accuracy':
            C = 10**float(sys.argv[3])
            eta = 10**float(sys.argv[4])
            T = int(sys.argv[5])
            print svm_sgd_calc_accuracy(C, eta, T)
        else:
            print "Error: please choose a valid command"
    elif sys.argv[1] == '7':
        pass
    else:
        print "Error: please choose a valid command"
