import sys
import q2

k_fold = int(sys.argv[1])
output_filename = sys.argv[2]

#Plot and save question 2f
q2.plot_2f(k_fold, output_filename)
