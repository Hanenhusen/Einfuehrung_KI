import numpy as np
from matplotlib import pyplot as plt
import os
os.system("cls")

# Einleitung
############
# Wir haben wieder eine Meßreihe, diesmal mit 100 Datenpunkten.
noise = 0.1
coords_x = np.linspace(0,6.28,100)
coords_y = 1.0 + 2.0 * np.sin(coords_x) + 3.0 * np.cos(coords_x)
coords_x = coords_x + np.random.normal(0.0, noise, coords_x.shape)
coords_y = coords_y + np.random.normal(0.0, noise, coords_y.shape)

# Aufgabe 2a
############
# Plotten Sie die Koordinaten in ein Koordinatensystem um sich einen 
# Überblick zu verschaffen

# plt.plot(coords_x, coords_y, "o")
# plt.show()

# Aufgabe 2b
############
# Schätzen Sie nun wie in Aufgabe 1 ein lineares Modell der Form
#
#   y = w0 + w1 * sin(x) + w2 * cos(x)
#
# Geben Sie die geschätzten Modellparameter auf der Konsole aus

# Definieren der Matrix X
X = np.stack([np.ones(coords_x.shape), np.sin(coords_x), np.cos(coords_x)], 1)

# Berechnen der Modellparameter mithilfe der Formel
XT = np.transpose(X) # Die Transponierte Matrix

w = np.linalg.inv(XT.dot(X)).dot(XT).dot(coords_y) # Berechnung der Modellparameter

# Ausgabe der geschätzten Modellparameter
print("Geschätzte Modellparameter:")
print("w0:", w[0])
print("w1:", w[1])
print("w2:", w[2])


# Aufgabe 2c
############
# Plotten Sie die durch ihre Modellparameter geschätzte Funktion in das gleiche Koordinatensystem

# Definieren der geschätzten Funktion
def estimated_function(x, w):
    return w[0] + w[1] * np.sin(x) + w[2] * np.cos(x)

# Plotten der geschätzten Funktion zusammen mit den Datenpunkten
plt.plot(coords_x, coords_y, "o", label="Datenpunkte")  # Plot der Datenpunkte
coords_x = np.linspace(0,7)
plt.plot(coords_x, estimated_function(coords_x, w), label="Geschätzte Funktion")  # Plot der geschätzten Funktion
plt.legend()  # Zeige die Legende für die beiden Plots
plt.show()
