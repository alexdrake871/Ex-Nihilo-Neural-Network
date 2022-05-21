import math
import random

def sin_data_set(data_points = 1000):
    '''Returns a dataset. X is a coordinate pair and y is a boolean int indicating whether or not the coordinate pair is above y = sin(x).'''
    X = []
    y = []
    for i in range(data_points):
        x_val = random.uniform(0.0, 2 * math.pi)
        y_val = random.uniform(-1.25, 1.25)
        X.append([x_val, y_val])
        y.append(int(y_val > math.sin(x_val)))
    
    return X, y


# dsX, dsy = sin_data_set(10)
# print(list(zip(dsX, dsy)))