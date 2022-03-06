# Base Layer

class Base:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_pass(self, input):
        """
        Function for Forward pass
        :param input: input to the layer
        """
        raise NotImplementedError

    def backward_pass(self, output_error, learning_rate):
        """
        Function for Backward Pass(Means Finding Gradients and Changing Weights)
        :param output_error: (predicted - true)
        :param learning_rate: learning_rate (rate of learning of network)
        """
        raise NotImplementedError

