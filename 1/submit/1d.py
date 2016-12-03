import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from knn import measure_knn

output_filename = sys.argv[1]

#Measure for n in 1..100
ns = range(100, 5000 + 1, 100)
results = {}
k=1
for n in ns:
    results[n] = measure_knn(k=k, n=n, verbose=True)

#Plot and save result
plt.plot(ns, [results[n] for n in ns], 'k.-')
plt.savefig(output_filename)
