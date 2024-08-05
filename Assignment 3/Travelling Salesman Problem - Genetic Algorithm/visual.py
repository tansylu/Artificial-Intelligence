import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('performance.txt')
plt.plot(data[:, 0], data[:, 1])
plt.xlabel('Generation')
plt.ylabel('Best Distance')
plt.title('Performance of the Population with Generations')
plt.show()