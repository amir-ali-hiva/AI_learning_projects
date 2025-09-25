import matplotlib.pyplot as plt
import numpy as np


f = lambda x: 4 * x + 2

start_data = -100
the_end = 100 

x = np.linspace (start_data, the_end , 100)

y = f(x)

plt.plot( x, y,"-b")
plt.show()
