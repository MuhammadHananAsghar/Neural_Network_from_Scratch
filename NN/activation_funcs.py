import numpy as np


# Sigmoid
def sigmoid(x):
    sigma = 1.0 / (1 + np.exp(-x))
    return sigma


def sigmoid_derivative(x):
    sigma = 1.0 / (1 + np.exp(-x))
    return sigma * (1 - sigma)


# Relu
def relu(x):
    return np.maximum(0.0, x)


def dRelu(x):
    return np.where(x <= 0, 0, 1)


def LeakyRelu(x):
    alpha = 0.1
    return np.maximum(alpha * x, x)


# LeakyRelu
def LeakyRelu_derivative(x):
    alpha = 0.1
    return np.where(x < 0, alpha*x, x)

# Tanh
def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2