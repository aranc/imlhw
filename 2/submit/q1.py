import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw2 import *



if True:
    #Get subquestion from first argument
    if sys.argv[1] == 'a':
        #get _from, _to, _step, and plot output filename from remaining arguments
        _from = float(sys.argv[2])
        _to = float(sys.argv[3])
        _step = float(sys.argv[4])
        output = sys.argv[5]
        q3a(_from, _to, _step, output)
    if sys.argv[1] == 'b':
        #get _from, _to, _step, and plot output filename from remaining arguments
        _from = float(sys.argv[2])
        _to = float(sys.argv[3])
        _step = float(sys.argv[4])
        output = sys.argv[5]
        q3b(_from, _to, _step, output)
    if sys.argv[1] == 'c':
        output = sys.argv[2]
        q3c(output)
    if sys.argv[1] == 'd':
        print q3d()
    else:
        print "Error: please choose subquestion (a,b,c,d)"
