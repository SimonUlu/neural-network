import numpy as np

class Loss:
    def __init__(self, type="mse"):
        self.type = type
        self.prediction = None
        self.target = None

    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        if self.type == "mse":
            return self.mse()
        elif self.type == "entropy":
            return self.entropy()
        else:
            raise ValueError("Unbekannter Loss-Typ")

    def backward(self):
        if self.type == "mse":
            return self.mse_derivative()
        elif self.type == "entropy":
            return self.entropy_derivative()
        else:
            raise ValueError("Unbekannter Loss-Typ")

    def mse(self):
        return np.mean((self.prediction - self.target) ** 2)
    
    def mse_derivative(self):
        return 2 * (self.prediction - self.target) / self.target.size

    def entropy(self):
        epsilon = 1e-12
        self.prediction = np.clip(self.prediction, epsilon, 1. - epsilon)
        return -np.sum(self.target * np.log(self.prediction)) / self.target.shape[0]
    
    def entropy_derivative(self):
        # Direkte Ableitung der Kreuzentropie-Kostenfunktion für die Mehrklassenklassifizierung
        return (self.prediction - self.target) / self.target.shape[0]