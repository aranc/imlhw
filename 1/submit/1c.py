import sys
import matplotlib.pyplot as plt
from knn import measure_knn

output_filename = sys.argv[1]

#Measure for k in 1..100
ks = range(1, 100 + 1)
results = {}
n=1000
for k in ks:
    results[k] = measure_knn(k=k, n=n, verbose=True)

#Plot and save result
plt.plot(ks, [results[k] for k in ks], 'k.-')
plt.savefig(output_filename)
