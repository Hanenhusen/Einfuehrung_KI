import torch
from torch import nn
from tqdm import tqdm
import os

# clear console
os.system("cls" if os.name == "nt" else "clear")

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Aufgabe 3:
###########
# Wir wollen nun das XOR-Problem in PyTorch lösen.
# Dazu sind folgende Daten gegeben
X = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]], device=device)
Y = torch.tensor([[0.0, 1.0, 1.0, 0.0]], device=device)

# Wir wollen ein zwei-schichtiges Netzwerk wie aus der Vorlesung
# bauen. In der Eingabeschicht gibt es zwei Neuronen (x1, x2), i
# in der versteckten Schicht sollen 4 Neuronen sein (h1,h2,h3,h4).
# Die Ausgabe benötigt ein Neuron (o1). Siehe unten.
#
#         h1
#
#     x1  h2
#             o1
#     x2  h3
#
#         h4
#
# Erzeugen Sie für die Gewichte der ersten und zweiten Schicht je eine
# Matrix. Sie können torch.rand verwenden um mit zufälligen Gewichten zu starten.
# Setzen Sie auf jedenfall requires_grad=True um die Gradietenberechnung zu aktivieren.
#
# Hinweis: Für die erste Schicht benötigen Sie eine (4,2)-Matrix. Für die zweite Schicht
# benötigen Sie eine (1,4)-Matrix.
W1 = torch.rand((4, 2), requires_grad=True, device=device)
W2 = torch.rand((1, 4), requires_grad=True, device=device)

# Wir verwenden eine Lernrate von 0.1 und zur Anzeige wieder einen TQDM-Bar
eta = 0.1
bar = tqdm(range(15000))

# Wir iterieren über den TQDM-Bar
for _ in bar:
    # Berechnen Sie die Netzeingabe Z und die Aktiverung O der Neuronen in der ersten Schicht
    # Dazu müssen Sie lediglich X mit W1 multiplizieren und das Ergebniss in die Sigmoid-Funktion stecken.
    Z1 = W1 @ X
    O1 = torch.sigmoid(Z1)

    # Berechnen Sie nun Netzeingabe Z und die Aktiverung O des Neurons in der Ausgabeschicht.
    # Dazu müssen Sie die Aktivierung der ersten Schicht mit W2 multiplizieren und das Ergebniss
    # in die Sigmoid-Funktion stecken.
    Z2 = W2 @ O1
    O2 = torch.sigmoid(Z2)

    # Da es sich im ein Klassifikationsproblem handelt müssen wir wieder wie bei der logistischen
    # Regression die Likelihood berechnen (vgl. Aufgabe 2)
    likelihood = torch.prod(torch.pow(O2, Y) * torch.pow(1.0 - O2, 1.0 - Y))

    # Um den Fortschritt des Trainings zu beobachten updaten wir den TQDM-Bar
    p = torch.pow(likelihood, 0.25).item()
    bar.set_description(
        f"likelihood={likelihood.item():.8f}, geometric mean={p*100.0:.2f}%"
    )

    # Führen Sie nun den Backward-Pass aus um die Gradienten von W1 und W2 zu berechnen
    likelihood.backward()

    # Updaten Sie wie in Aufgabe 2 die Gewichte und setzen Sie anschließend die Gradienten auf 0
    # zurück
    W1.data = W1.data + eta * W1.grad
    W2.data = W2.data + eta * W2.grad
    W1.grad.data.zero_()
    W2.grad.data.zero_()


# Geben Sie die Ausgaben für ihr letztes Layer nach dem Training aus um ihr Ergebniss zu überprüfen
print("\nErgebnis:")
print(O2)
print("Erwartet:")
print(Y)
