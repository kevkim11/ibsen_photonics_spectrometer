import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_and_filter import plot_dyes
from os.path import join
from k_baseline import line_scanner, K_Baseline

folder = '../csv_files'
folder_name = 'baseline_subtraction.csv'

file_dir = join(folder, folder_name)

a = pd.read_csv(file_dir, index_col=0)

data = a.transpose()
cxr = data.iloc[0].tolist()

flu = []
joe = []
tmr = []
wen = []
l_of_l = [flu, joe, tmr, cxr, wen]

# Initialize
print cxr
# x = np.array([0,2,4])
# y = np.array([0,15,0])
#
# z = [y, x]



# p1 = plot_dyes(l1, scatter=True)
# p1.set_title("dummy")
# p1.grid(True)
#
# plt.show()

