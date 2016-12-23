import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.svm
from hw2 import *

#Measure SVM classifier on a data set
def measure(classifier, data, labels):
    errors = 0
    for i in range(len(data)):
        predicted = S.predict(data[i].reshape(1, -1))
        if predicted != labels[i]:
            errors += 1
    return 1 - float(errors)/float(len(data))

#Plot for question 2a
def q2a(_from, _to, _step, output=None):
    #Measure for a specific C value
    def measure_for_C(C):
        S=sklearn.svm.LinearSVC(loss='hinge', fit_intercept=False, C=C)
        S.fit(train_data, train_labels)
        return measure(S, train_data, train_labels), measure(S, train_data, validation_labels)

    p_range = np.arange(_from, _to, _step)
    training_accuracy = {}
    validation_accuracy = {}
    for p in p_range:
        C = 10**p
        training_accuracy[p], validation_accuracy[p] = measure_for_C(C)
        print "p:", p, "training accuracy:", training_accuracy[p], "validation accuracy:", validation_accuracy[p]
    plt.gca().set_xlabel("log10(C)")
    plt.gca().set_ylabel("Accuracy")
    plt.plot(p_range, [validation_accuracy[p] for p in p_range], 'ko', label="validation")
    plt.plot(p_range, [training_accuracy[p] for p in p_range], 'k*', label="training")
    plt.legend()
    if output == None:
        plt.show()
    else:
        plt.savefig(output)


if True:
    #Get subquestion from first argument
    if sys.argv[1] == 'a':
        #get _from, _to, _step, and plot output filename from remaining arguments
        _from = float(sys.argv[2])
        _to = float(sys.argv[3])
        _step = float(sys.argv[4])
        output = sys.argv[5]
        q2a(_from, _to, _step, output)
    else:
        print "Error: please choose subquestion (a,b,c,d)"
