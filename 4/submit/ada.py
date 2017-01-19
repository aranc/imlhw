import sys
import time
import operator
import math
from math import e
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw4 import *

#Build weak learner from parameters
def build_weak_learner(pixel, theta, direction):
    if direction:
        return lambda x: -1 if x[pixel] <= theta else 1
    else:
        return lambda x: 1 if x[pixel] <= theta else -1

#Measure error of classifier
def measure(classifier, data, labels):
    errors = 0
    for i in range(len(data)):
        if labels[i] != classifier(data[i]):
            errors += 1
    return float(errors) / float(len(data))

#Measure error of classifier, with a given distribution
def measure_with_distribution(classifier, data, labels, distribution):
    errors = 0.0
    for i in range(len(data)):
        if labels[i] != classifier(data[i]):
            errors += distribution[i]
    return errors

#Return the best weak learner for the given distribution
def find_best_weak_learner(data, labels, distribution):
    #Consts
    pixels = len(data[0])

    #Parameters to argmin
    best_pixel = 0
    best_theta = 0
    best_direction = False
    best_error = float("inf")

    #Find argmin error
    start = time.time()
    for pixel in range(pixels):
        possible_thetas = np.unique(data[:,pixel])
        for theta in possible_thetas:
            for direction in (True, False):
                learner = build_weak_learner(pixel, theta, direction)
                error = measure_with_distribution(learner, data, labels, distribution)
                if error < best_error:
                    best_error = error
                    best_direction = direction
                    best_theta = theta
                    best_pixel = pixel

    print "find_best_weak_learner: elapsed:", time.time() - start, "error:", best_error, "pixel:", best_pixel, "theta:", best_theta, "direction:", best_direction

    return build_weak_learner(best_pixel, best_theta, best_direction)


#Perform an additional iteration of adaboost
#Modifies distribution, and returns h_t and a_t
def adaboost_iteration(data, labels, distribution):
    #Recive h_t
    h_t = find_best_weak_learner(data, labels, distribution)
    
    #Define error and alpha
    e_t = measure_with_distribution(h_t, data, labels, distribution)
    a_t = (1.0 / 2.0) * math.log( (1 - e_t) / e_t)

    #Define D_t+1
    for i in range(len(distribution)):
        distribution[i] *= (e**(-a_t) if labels[i] == h_t(data[i]) else e**(a_t))

    #Normalize
    z_t = np.sum(distribution)
    distribution /= z_t

    return a_t, h_t

#Build linear classifier from lists of a_t and h_t
def build_linear_classifier(a, h):
    def classifier(x):
        _sum = 0
        for i in range(len(a)):
            _sum += a[i] * h[i](x)
        return -1 if _sum < 0 else 1
    return classifier


#Calculate average exponential loss
def calc_average_exponential_loss(a, h, data, labels):
    m = len(data)
    T = len(h)

    ael_error = 0
    for i in range(m):
        exponent = 0
        for j in range(T):
            exponent += a[j] * h[j](x[i])
        exponent *= -y[i]
        ael_error += ( e**exponent ) / m

    return ael_error


#Run T iterations of adaboost, prepare plots for subquestions a and b
def answer_subquestions(T, plot_a, plot_b):
    #Init uniform distribution
    distribution = np.ones(len(train_data)) / len(train_data)

    #List of h_t
    h = []
    #List of a_t
    a = []
    #List for the plots
    training_errors = []
    test_errors = []
    ael_training_errors = []
    ael_test_errors = []

    #Run T iterations
    for t in range(T):
        a_t, h_t = adaboost_iteration(train_data, train_labels, distribution)
        a.append(a_t)
        h.append(h_t)
        classifier = build_linear_classifier(a, h)
        training_errors.append(measure(classifier, train_data, train_labels))
        test_errors.append(measure(classifier, test_data, test_labels))
        ael_training_error.append(calc_average_exponential_loss())
        ael_test_error.append(calc_average_exponential_loss())
        print "t:", t, "training error:", training_errors[-1], "testing_error:", test_errors[-1]

    #Prepare plots
    plt.title("Training and Test errors")
    plt.gca().set_xlabel("T iterations")
    plt.gca().set_ylabel("Error")
    plt.plot(range(1,T+1), test_errors, 'ko', label="Test Error")
    plt.plot(range(1,T+1), training_errors, 'k*', label="Training Error")
    plt.legend()
    plt.savefig(plot_a)

    plt.clf()
    plt.title("Training and Test average exponential loss")
    plt.gca().set_xlabel("T iterations")
    plt.gca().set_ylabel("AEL")
    plt.plot(range(1,T+1), ael_test_errors, 'ko', label="Test AEL")
    plt.plot(range(1,T+1), ael_training_errors, 'k*', label="Training AEL")
    plt.legend()
    plt.savefig(plot_a)

if True:
    T = int(sys.argv[1])
    plot_a = sys.argv[2]
    plot_b = sys.argv[3]
    answer_subquestions(T, plot_a, plot_b)
