import numpy as np
data = np.genfromtxt('output.csv', delimiter=',', names=['t','x'])
import matplotlib as mpl
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data['t'],data['x'],color='r',label='position')
plt.show()

