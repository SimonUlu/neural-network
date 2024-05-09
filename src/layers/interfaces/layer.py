class Layer():
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_gradient):
        raise NotImplementedError
