#test LinearRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: 4 * x + 2
start = -10
end = 10
data = 50

x = np.linspace(start, end, data)

y = f(x)

nois = np.random.randn(data) * 5
y_nois = y + nois

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
y_nois = y_nois.reshape(-1, 1)


modle = LinearRegression()
modle.fit(x, y_nois)

g = lambda x : modle.coef_ * x + modle.intercept_

y_predictc = g (x)

plt.plot(x ,y_predictc , "sk")
plt.plot(x , y, "-r")
plt.plot(x, y_nois,"ob")
plt.legend(["predictc", "Orginaly line", "data nois"])
plt.title(f"y= 4 *x + 2  y_hat = {modle.coef_} * x + {modle.intercept_} ")
plt.show()