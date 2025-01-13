import numpy as np
from matplotlib import pyplot as plt
import os
os.system("cls")

from matplotlib import pyplot as plt

# Einleitung
############
# Wir haben eine Meßreihe mit 4 Datenpunkten aufgenommen und gemessen
#
#      x | 1 | 2 | 3 | 4
#       -----------------
#      y | 3 | 4 | 4 | 2
#
# Wir möchten ein linears Modell der Form
#
#   y = w0 + w1*x + w2*x² + w3x³ 
#
# schätzen und dazu die vier Parameter w0, w1, w2 und w3 bestimmen.

# Aufgabe 1a
############
# Definieren Sie geeignte NumPy Arrays um die gegebenen Daten verwenden zu können.
# Verwenden Sie die Funktionen np.ones_like und np.stack um die aus der Vorlesung bekannte
# X-Matrix für das gegebenen Modell aufzustellen.

x = np.array([1, 2, 3, 4])
y = np.array([3, 4, 4, 2])
X = np.stack([np.ones_like(x), x, x**2, x**3], 1)

print("")
print(X)
# Aufgabe 1b
############
# Bestimmen Sie nun mittel np.linalg.inv die Pseudeoinverse von X sowie die Modellparameter w0,w1,w2 und w3
# Geben Sie diese auf der Konsole aus. Berechnen Sie ebenfalls
# den summarischen quadratischen Fehler ihrer Vorhersage und geben Sie auch diese auf der Konsole aus.

model = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y

print("")
print(f"w0= {model[0]}; w1= {model[1]}; w2= {model[2]}; w3= {model[3]}")

def model_function(x):
    return model[0] + model[1]*x + model[2]*x**2 + model[3]*x**3

summe_quadratischen_fehler = np.sum((y - model_function(x))**2) / len(x)


# for i in range(len(x)):
#     summe_quadratischen_fehler += (y[i] - model_function(x[i]))**2

print("")
print(f"Summe Quadratischer Fehler: {summe_quadratischen_fehler}")

# Aufgabe 1c
############
# Plotten Sie die Meßreihe sowie das geschätzte Modell 
# plot punkte
plt.plot(x, y, "o")
# plot model
plt.plot(np.linspace(x[0], x[-1], 200), model_function(np.linspace(x[0], x[-1], 200)))
plt.show()

# Aufgabe 1d
############
# Was verändert sich wenn Sie ihrer Meßreihe weitere Koordinaten hinzufügen, z.B den
# Punkt (5,2)?

# Fehler wird größer und die Punkte sind weiter von der Kurve entfernt
