from base import Base
import numpy as np


class Activation(Base):
    def __init__(self, activation, activation_derivative):
        # activation = activation which we used for FORWARD PASS
        # activation_derivative = derivative of activation used for BACKWARD PASS
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward_pass(self, _input):
        self.input = _input
        self.output = self.activation(self.input)
        return self.output

    def backward_pass(self, output_error, learning_rate):
        """
        In this backward pass we are to return
        (de/dx)
        where de/dx = f`(X) * de/dy
        """
        return self.activation_derivative(self.input) * output_error