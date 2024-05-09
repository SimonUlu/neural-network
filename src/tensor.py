import numpy as np

class Tensor:
    def __init__(self, elements, shape):
        self.elements = np.array(elements)
        self.deltas = np.zeros_like(self.elements)
        self.shape = shape

        
class Shape:
    def __init__(self, *dims):
        self.dims = dims