from network import NeuralNetwork
from linear import Linear
from activation import Activation
from loss import mean_square_error, mean_squared_error_derivative
from activation_funcs import relu, dRelu, sigmoid, sigmoid_derivative
from normalization import normalize
from sklearn.datasets import make_classification
import numpy as np

# Random Classification Data
X, y = make_classification(n_samples=200, n_features=5, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)

m, n = X.shape
# Reshaping y.
y = y.reshape(m , 1)
# Normalizing
x = normalize(X)
x = np.expand_dims(x, axis=1)

x_train, x_test, y_train, y_test = x[:180], x[180:], y[:180], y[180:]

nn = NeuralNetwork()
nn.add(Linear(input_dim=5, output_dim=20))
nn.add(Activation(relu, dRelu))
nn.add(Linear(input_dim=20, output_dim=1))
nn.add(Activation(sigmoid, sigmoid_derivative))
nn.ready(mean_square_error, mean_squared_error_derivative)

nn.run(x_train=x_train, y_train=y_train, epochs=200, lr=0.1)

predictions = nn.predict(x_test)
values = [1 if i > 0.5 else 0 for i in predictions]
print(values, y_test.reshape(1, -1))