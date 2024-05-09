from src.layers.interfaces.layer import Layer
from src.tensor import Tensor, Shape

class InputLayer(Layer):
    def forward(self, input):
        return Tensor(input, Shape(*input.shape))
    
    def backward(self, output_gradient):
        return output_gradient