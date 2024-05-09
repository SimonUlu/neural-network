from src.layers.interfaces.layer import Layer
import numpy as np

class FullyConnected(Layer):
    def __init__(self, input_size, output_size):
        stddev = np.sqrt(2. / input_size)
        self.weights = np.random.normal(0, stddev, (input_size, output_size))
        self.biases = np.full(output_size, 0.10)  # Alternative mit kleinen konstanten Werten  # Biases können mit 0 initialisiert werden
        self.input = None
        self.weights_gradient = None
        self.biases_gradient = None
    
    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, output_gradient):
        # Berechnung des Gradienten der Gewichte und Biases
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.biases_gradient = np.mean(output_gradient, axis=0)
        
        # Berechnung des Gradienten, der an die vorherige Schicht zurückgegeben wird
        return np.dot(output_gradient, self.weights.T)
    
    def calculate_delta_weights(self, learning_rate):
        # Berechnung der Delta-Gewichte und Delta-Biases
        delta_weights = -learning_rate * self.weights_gradient
        delta_biases = -learning_rate * self.biases_gradient
        
        return delta_weights, delta_biases
    
    def update_weights(self, delta_weights, delta_biases):
        # Aktualisierung der Gewichte und Biases mit den berechneten Deltas
        self.weights += delta_weights
        self.biases += delta_biases