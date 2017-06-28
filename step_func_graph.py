import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    # xの各要素が０以上かどうかで、結果をnp.int
    return np.array(x > 0,dtype = np.int)

x = np.arange(-5.0,5.0,0.1)
print(x)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
