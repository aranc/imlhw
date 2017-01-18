import sys
import time
import operator
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw4 import *

if True:
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
