from abc import ABC, abstractmethod
import numpy as np

class ActivationStrategy(ABC):

    @staticmethod
    @abstractmethod
    def activation(x):
        pass

    @staticmethod
    @abstractmethod
    def derivative(x):
        pass


class SigmoidStrategy(ActivationStrategy):

    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        sigmoid_x = SigmoidStrategy.activation(x)
        return sigmoid_x * (1 - sigmoid_x)

class ReluStrategy(ActivationStrategy):

    @staticmethod
    def activation(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)

class SoftmaxStrategy(ActivationStrategy):

    @staticmethod
    def activation(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def derivative(output):
        pass