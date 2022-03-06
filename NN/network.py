class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def add(self, layer):
        self.layers.append(layer)

    def ready(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    def predict(self, _input):
        samples = len(_input)
        result = []

        for idx in range(samples):
            output = _input[idx]
            for layer in self.layers:
                output = layer.forward_pass(output)

            result.append(output)
        return result

    def run(self, x_train, y_train, epochs, lr):

        samples = len(x_train)
        for epoch in range(epochs):
            _loss = 0

            for idx in range(samples):
                output = x_train[idx]

                for layer in self.layers:
                    output = layer.forward_pass(output)

                _loss = _loss + self.loss(output, y_train[idx])

                _loss_derivative = self.loss_derivative(output, y_train[idx])

                for layer in reversed(self.layers):
                    _loss_derivative = layer.backward_pass(_loss_derivative, lr)

            mean_loss = _loss/samples

            print(f"[*] Epoch:- {epoch}/{epochs} -> Error = {mean_loss}")