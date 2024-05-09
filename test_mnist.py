from src.network import Network
from src.layers.fully_connected import FullyConnected
from src.layers.interfaces.activation_strategy import SigmoidStrategy, ReluStrategy, SoftmaxStrategy
from src.layers.activation import ActivationLayer
from src.layers.input import InputLayer
from src.layers.loss import Loss
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# MNIST-Datensatz laden und vorbereiten
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[:5000]
train_labels = train_labels[:5000]

test_images = test_images[:1000]
test_labels = test_labels[:1000]

# Normalisieren der Pixelwerte von [0, 255] auf [0, 1]
train_images = train_images.reshape((5000, 784)).astype('float32') / 255
test_images = test_images.reshape((1000, 784)).astype('float32') / 255

# Labels in One-Hot-Encoding umwandeln
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Netzwerkarchitektur anpassen
network = Network()
network.add(InputLayer())
network.add(FullyConnected(input_size=784, output_size=16))
network.add(ActivationLayer(ReluStrategy()))
network.add(FullyConnected(input_size=16, output_size=24))
network.add(ActivationLayer(ReluStrategy()))
network.add(FullyConnected(input_size=24, output_size=12))
network.add(ActivationLayer(ReluStrategy()))
network.add(FullyConnected(input_size=12, output_size=10))  # Anpassung für 10 Ausgabeneuronen
network.add(ActivationLayer(SigmoidStrategy()))  # Softmax für Mehrklassenklassifizierung

network.set_loss(Loss(type="entropy"))  # Verlustfunktion für Mehrklassenklassifizierung

learning_rate = 0.05

# Trainingsschleife anpassen
losses = []
for epoch in range(15):  # Nehmen wir an, wir trainieren für 50 Epochen
    epoch_loss = 0
    for x_train, y_train in zip(train_images, train_labels):
        x_train = x_train.reshape(1, 784)  # Anpassen der Form für das Netzwerk
        y_train = y_train.reshape(1, 10)
        prediction = network.forward(x_train)
        loss = network.compute_loss(prediction, y_train)
        network.backprop(y_train)
    
        # Gewichte und Biases aktualisieren
        for layer in network.layers:
            if isinstance(layer, FullyConnected):
                delta_weights, delta_biases = layer.calculate_delta_weights(learning_rate)
                layer.update_weights(delta_weights, delta_biases)
        epoch_loss += loss
    losses.append(epoch_loss / len(train_images))

# Ausgabe der Verlustentwicklung
print("Verlustentwicklung:", losses[-10:])  # Die letzten 10 Verlustwerte anzeigen

# Genauigkeit berechnen (Sie müssen die calculate_accuracy Funktion entsprechend anpassen)
accuracy = network.calculate_accuracy(network, test_images, test_labels)
print("Testgenauigkeit:", accuracy)