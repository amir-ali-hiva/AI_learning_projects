#hello AI
print("test")
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# def f(x):
#     return 4 * x + 10
f = lambda x: 4 * x + 10

number_of_data = 50
X = np.linspace(-100, 100, number_of_data)  # X = np.linspace(-100, 100, number_of_data).reshape(-1, 1)
Y = f(X)
noise = np.random.randn(number_of_data) * 20
Y_Noised = Y + noise

X = X.reshape(-1, 1)  # shape: (50,) => shape: (50, 1)
Y = Y.reshape(-1, 1)
Y_Noised = Y_Noised.reshape(-1, 1)
# داخل سایکیت لرن، داده های ورودی باید دو بعدی باشند
model = LinearRegression()
model.fit(X, Y_Noised)
# a_hat = model.coef_, b_hat = model.intercept_
g = lambda x: model.coef_ * x + model.intercept_
Y_Predicted = g(X)

plt.plot(X, Y, "-b")
plt.plot(X, Y_Noised, "or")
plt.plot(X, Y_Predicted, "sg")
plt.plot(X, Y_Predicted, "-k")
plt.title(f"Y = 4 * X + 10, Y_hat = {model.coef_} X + {model.intercept_}")
# باید به ترتیبی که پلات کردی، توضیحشم داخل لجند بگذاری
plt.legend(["Original Line", "Data with Noise", "Pridected Data", "Predicted Line"])
plt.show()