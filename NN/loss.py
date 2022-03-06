import numpy as np


# MSE - Mean Squared Error
def mean_square_error(y_pred, y_true):
    output = np.mean(np.power(y_pred - y_true, 2))
    return output


# MSE - Derivative
def mean_squared_error_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size
