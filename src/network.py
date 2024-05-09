import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.loss_function = None

    def add(self, layer):
        self.layers.append(layer)
    
    def set_loss(self, loss_function):
        self.loss_function = loss_function
    
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def compute_loss(self, prediction, target):
        return self.loss_function.forward(prediction, target)
    
    def backprop(self, target):
        # Berechne den Gradienten des Verlusts
        loss_gradient = self.loss_function.backward()
        # Führe Backpropagation durch das Netzwerk durch
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)
        return loss_gradient
    
    def calculate_accuracy(self, network, data, labels):
        correct_predictions = 0
        for x, y in zip(data, labels):
            x = x.reshape(1, 784)  # Anpassen der Eingabe
            prediction = network.forward(x)
            if np.argmax(prediction) == np.argmax(y):
                correct_predictions += 1
        return correct_predictions / len(data)