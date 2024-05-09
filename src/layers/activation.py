from src.layers.interfaces.layer import Layer

class ActivationLayer(Layer):
    def __init__(self, activation_strategy):
        self.activation_strategy = activation_strategy
        self.input = None

    def forward(self, input):
        self.input = input
        return self.activation_strategy.activation(input)

    def backward(self, output_gradient):
        return output_gradient * self.activation_strategy.derivative(self.input)