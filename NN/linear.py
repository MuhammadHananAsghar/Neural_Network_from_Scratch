import numpy as np
from base import Base


# Linear Layer

class Linear(Base):
    # input_dim = number of input neurons
    # output_dim = number of output neurons
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(input_dim, output_dim) - 0.6
        self.bias = np.random.rand(1, output_dim) - 0.6

    def forward_pass(self, _input):
        self.input = _input
        # y = wx + b
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_pass(self, output_error, learning_rate):
        """
        Backpropogation means: Find partial derivative
        e.g: w<-y<-e
        where w is the weight of neuron
        where y is the output of neuron
        where e is the total error (y - real)
        so by using Chain Rule
        de/dw = de/dy * dy/dw (For one Neuron)
        """

        # dw = dz * x.T(Transpose of x)
        # where dz = activation_output - real_output (Means output error)

        # Finding Gradients
        dw = np.dot(self.input.T, output_error)
        # Changing Gradients (Updating Gradients)
        self.weights = self.weights - (learning_rate * dw)
        self.bias = self.bias - (learning_rate * output_error)

        # Find Error of Input w.r.t Current Weights
        input_error = np.dot(output_error, self.weights.T)

        return input_error
