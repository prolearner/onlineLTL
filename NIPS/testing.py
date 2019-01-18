import numpy as np
import matplotlib.pylab as plt

n_dims = 1
n_points = 100

w = [-13, 44]

x = np.linspace(-100, 100, n_points).reshape(n_points, 1)

X = np.ones((n_points, n_dims + 1))
X[:, :-1] = x

Y = X @ w

plt.plot(X[:, 0], Y)
plt.pause(0.01)
k=1