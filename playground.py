import numpy as np
import matplotlib.pyplot as plt
from plot_and_filter import plot_dyes

x = np.array([1,2,3])
y = np.array([1,2,3])


z = [x, y]

p1 = plot_dyes(z, scatter=True)
p1.set_title("dummy")
p1.grid(True)

plt.show()

