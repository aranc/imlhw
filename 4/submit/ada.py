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

#Return the best weak learner for the given distribution
def find_best_weak_learner(data, labels, distribution):
    #Consts
    pixels = 28

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
                error = measure(learner, data, labels)
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
    e_t = measure(h_t, data, labels)
    a_t = (1.0 / 2.0) * math.log( (1 - e_t) / e_t)

    #Define D_t+1
    for i in range(len(distribution)):
        distribution[i] *= e**(-a_t) if labels[i] == h_t(data[i]) else e**(a_t)

    #Normalize
    z_t = np.sum(distribution)
    distribution /= z_t


if False:
    if len(sys.argv) > 1 and sys.argv[1] == 'bla':
        _from = float(sys.argv[3])
        _to = float(sys.argv[4])
        _step = float(sys.argv[5])
        C = 10**float(sys.argv[6])
        T = int(sys.argv[7])
        filename = sys.argv[8]
        svm_sgd_find_eta(_from, _to, _step, C, T, filename)
    else:
        print "Error: please choose a valid command"
