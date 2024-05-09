from src.network import Network
from src.layers.fully_connected import FullyConnected
from src.layers.interfaces.activation_strategy import SigmoidStrategy, ReluStrategy, SoftmaxStrategy
from src.layers.activation import ActivationLayer
from src.layers.loss import Loss
import numpy as np


# Netzwerk initialisieren
network = Network()

# FullyConnected Layer hinzufügen
network.add(FullyConnected(input_size=2, output_size=16))  # Beispiel: Eingabe von 784, Ausgabe von 100

# Aktivierungslayer hinzufügen, der die ReLU-Aktivierung verwendet
network.add(ActivationLayer(ReluStrategy()))

network.add(FullyConnected(input_size=16, output_size=24))  # Beispiel: Eingabe von 784, Ausgabe von 100

# Aktivierungslayer hinzufügen, der die ReLU-Aktivierung verwendet
network.add(ActivationLayer(ReluStrategy()))

network.add(FullyConnected(input_size=24, output_size=12))  # Beispiel: Eingabe von 784, Ausgabe von 100

# Aktivierungslayer hinzufügen, der die ReLU-Aktivierung verwendet
network.add(ActivationLayer(ReluStrategy()))

# FullyConnected Layer hinzufügen
network.add(FullyConnected(input_size=12, output_size=1))

# Angenommen, Sie möchten auch einen Sigmoid-Aktivierungslayer hinzufügen
network.add(ActivationLayer(SigmoidStrategy()))

network.set_loss(Loss(type="mse"))

# XOR Trainingsdaten
train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_targets = np.array([[0], [1], [1], [0]])

learning_rate = 0.05

# Trainingsschleife
losses = []
for epoch in range(50):  # Nehmen wir an, wir trainieren für 1000 Epochen
    epoch_loss = 0
    for x_train, y_train in zip(train_data, train_targets):
        x_train = x_train.reshape(1, 2)  # Anpassen der Form für das Netzwerk
        y_train = y_train.reshape(1, 1)
        prediction = network.forward(x_train)
        loss = network.compute_loss(prediction, y_train)
        network.backprop(y_train)
    
        #  Aktualisieren Sie die Gewichte und Biases für jede Schicht
        for layer in network.layers:
            if isinstance(layer, FullyConnected):
                delta_weights, delta_biases = layer.calculate_delta_weights(learning_rate)
                layer.update_weights(delta_weights, delta_biases)
        
        epoch_loss += loss
    losses.append(epoch_loss / len(train_data))

# Ausgabe der Verlustentwicklung
print("Verlustentwicklung:", losses[-50:])  # Die letzten 10 Verlustwerte anzeigen

print(network)

# Testen des trainierten Netzwerks
for x_test in train_data:
    x_test = x_test.reshape(1, 2)  # Anpassen der Form für das Netzwerk
    prediction = network.forward(x_test)
    print(f"Eingabe: {x_test} Vorhersage: {prediction}")