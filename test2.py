import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_diagram(X, Y, w1, w2, b1, b2):
    predictions = []
    for i in range(X.shape[0]):
        a0 = X[i]
        n1 = np.dot(w1, a0) + b1
        a1 = sigmoid(n1)
        n2 = np.dot(w2, a1) + b2
        a2 = linear(n2)
        predictions.append(a2)

    predictions = np.array(predictions).reshape(number_of_data,)
    mse = mean_squared_error(predictions, Y)
    mae = mean_absolute_error(predictions, Y)

    plt.clf()
    plt.plot(X, Y, "b-")
    plt.plot(X, Y, "or")
    plt.plot(X, predictions, "sg")
    plt.title(f"MAE: {mae}, MSE: {mse}")
    plt.legend(["Original Function", "Original Data", "Predicted Data"])
    plt.pause(0.001)

sigmoid = lambda x: 1 / (1 + np.exp(-1 * x))
d_sigmoid = lambda x: (1 - sigmoid(x)) * sigmoid(x)

linear = lambda x: x
d_linear = 1

g = lambda p : 1 + np.sin(np.pi / 4 * p)

w1 = np.array([-0.27, -0.41]).reshape(2, 1)
b1 = np.array([-0.48, -0.13]).reshape(2, 1)

w2 = np.array([0.09, -0.17]).reshape(1, 2)
b2 = np.array([0.48]).reshape(1, 1)

alpha = 0.1

var_min = -2
var_max = 2
number_of_data = 50
number_of_epochs = 100
X = np.linspace(var_min, var_max, number_of_data)
Y = g(X)

plt.figure()
# Training Phase
for j in range(number_of_epochs):   # چند دور آموزش # epochs
    #for i in range (j):
    for i in range(X.shape[0]):     # (یک دور آموزش)یکبار دادن تمامی داده ها به مدل
        a0 = X[i]
        n1 = np.dot(w1, a0) + b1
        a1 = sigmoid(n1)
        n2 = np.dot(w2, a1) + b2
        a2 = linear(n2)
        t = Y[i]
        e = t - a2
        s2 = -2 * d_linear * e
        F1 = np.array([[(d_sigmoid(n1[0,0])),0],[0,(d_sigmoid(n1[1,0]))]]).reshape(2, 2)
        s1 = np.dot(F1, w2.T) * s2
        w2 = w2 - alpha * s2 * a1.T
        b2 = b2 - alpha * s2
        w1 = w1 - alpha * s1 * a0
        b1 = b1 - alpha * s1
        plot_diagram(X, Y, w1, w2, b1, b2)

plt.show()