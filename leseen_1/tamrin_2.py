from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def mack_data(number_of_data = 100):
    
    f = lambda x: 4 * x + 5
    X = np.linspace(0, 100, number_of_data) 
    Y = f(X).reshape(-1, 1)
    X = X.reshape(-1, 1)
   
    Y_Noised = Y + np.random.randn(number_of_data,1) * 20

    x_train , x_test , y_train , y_test = train_test_split(X , Y_Noised , test_size= 0.2 , random_state= 85)

    return Y_Noised , X , Y , x_train , x_test , y_train , y_test

Y_Noised, X , Y , x_train , x_test , y_train , y_test = mack_data()



model = LinearRegression()
model.fit(x_train, y_train)


y_predict = model.predict(x_test)
# g = lambda x: model.coef_ * x + model.intercept_

# plt.plot(X, Y, "-b")
plt.plot(x_train, y_train, "or")
# plt.plot(X, Y_Predicted, "sg")
# plt.plot(X, Y_Predicted, "-k")
# plt.title(f"Y = 4 * X + 10, Y_hat = {model.coef_} X + {model.intercept_}")

# plt.legend(["Original Line", "Data with Noise", "Pridected Data", "Predicted Line"])
plt.show()