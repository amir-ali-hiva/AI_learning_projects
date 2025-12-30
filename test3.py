import matplotlib.pyplot as plt
import numpy as np

hardlim = lambda x: 1 if x > 0 else 0

def draw_line(W, P):
    slope = (W[0,1] - 0) / (W[0,0] - 0)
    g = lambda x: -1 / slope * x

    x = np.linspace(-2, 2, 10)
    y = g(x)

    plt.clf()
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.plot(x, y, "-b")
    plt.scatter(P[:, 0], P[:, 1], c=[1,0,0])
    plt.pause(2)
    
plt.figure()


P = np.array([[1,2],[-1,2],[0,-1]])
W = np.array([1, -0.8]).reshape(1, -1)
draw_line(W, P)

p = P[0,:].reshape(-1,1)
n = np.dot(W, p) 
a = hardlim(n)
W = W + p.T
draw_line(W, P)

p = P[1,:].reshape(-1,1)
n = np.dot(W, p)
a = hardlim(n)
W = W - p.T
draw_line(W, P)

p = P[2,:].reshape(-1,1)
n = np.dot(W, p)
a = hardlim(n)
W = W - p.T
draw_line(W, P)

plt.show()
