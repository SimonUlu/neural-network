import numpy as np

class ConvolutionLayer(Layer):
    def __init__(self, num_filters, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size) / (kernel_size * kernel_size)


    def iterate_regions(self, image):
        """
        Generator, der über die Bildbereiche iteriert, auf die die Faltung angewendet wird.
        """
        h, w = image.shape
        for i in range(h - self.kernel_size + 1):
            for j in range(w - self.kernel_size + 1):
                im_region = image[i:(i + self.kernel_size), j:(j + self.kernel_size)]
                yield im_region, i, 
    

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.kernel_size + 1, w - self.kernel_size + 1, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output
    
    def backward(self, output_gradient):
        raise NotImplementedError