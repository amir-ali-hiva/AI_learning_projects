from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np



def create_data( data_start = 0 , data_end = 100, number_of_data = 100 ):
    y = (lambda x : 3 * x + 4).reshape(-1 , 1)
    x = np.linspace(data_start, data_end ,number_of_data  ).reshape(-1 , 1)
    y_noize = y + np.random.rand(number_of_data , 1) * 20 

    x_train , x_test , y_train , y_nest = train_test_split(x , y_noize , test_size=0.2 , random_state= 69)


    return x_train , x_test , y_train , y_nest 





# round()