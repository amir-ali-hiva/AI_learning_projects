import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

f = lambda x : 2 * x + 5

number_of_data = 20

x = np.linspace (0 , 100 , number_of_data)
y = f(x)

y_noiss = np.random.rand(number_of_data, 1) * 10 + y
