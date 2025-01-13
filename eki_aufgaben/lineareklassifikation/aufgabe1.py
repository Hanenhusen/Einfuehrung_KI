import numpy as np
from matplotlib import pyplot as plt

# Einleitung
# #########
# In einem Zwei-Klassenproblem habe wir folgende Punkte gemessen
# 
# Rot  (1,3), (2,2), (3,3), (3,2), (2,4)
# Blau (3,5), (5,3), (5,4), (6,5), (4,4)

# Aufgabe 1a
# ##########
# Definieren Sie mit NumPy ein coords_x1 und coords_x2 Array für die gegebenen Punkte
# Definieren Sie weiterhin ein class_y Array für die Klassen der gegebenen Punkte (-1 und 1)
# Verwenden Sie dann plt.plot um die roten und blauen Punkte in ihrer jeweiligen Farbe zu zeichnen. 
coords_x1 = np.array([1,2,3,3,2,3,5,5,6,4])
coords_x2 = np.array([3,2,3,2,4,5,3,4,5,4])
class_y = np.array([-1,-1,-1,-1,-1,1,1,1,1,1])
plt.scatter(coords_x1[class_y == -1], coords_x2[class_y == -1], color = "Red")
plt.scatter(coords_x1[class_y == 1], coords_x2[class_y == 1], color = "blue")



# ##########
# Aufgabe 1b
# ##########
# Schätzen Sie nun wie in der vorherigen Aufgaben ein linears Modell der Form
#
#   y = w0 + w1 * x1 + w2 * x2 (Gl. 1)
#
# indem Sie die X-Matrix aus der Vorlesung aufstellen, die Pseudoinverse bestimmen und
# dann die Modellparameter bestimmen. 
X = np.stack([np.ones_like(coords_x1), coords_x1, coords_x2**2], 1)

pseudoinv = np.linalg.pinv(X)
model_param = pseudoinv @ class_y

print(f"Pseudoinverse : {pseudoinv}")
print("Parameter : ", model_param)

# Aufgabe 1c
# ##########
# Mit dem Modell aus Aufgabe 1c (Gl. 1) ist die Grenzfläche zwischen beiden Klassen 
# durch die Gleichung
#
#   y = w0 + w1 * x1 + w2 * x2 = 0 (Gl. 2)
#
# definiert. Lösen Sie Gl. 2 nach x2 auf und zeichen Sie diese Trennebene ebenfalls ein
x1_values = np.linspace(coords_x1.min(), coords_x1.max(), 400)
x2_values = -(model_param[1] * x1_values + model_param[0]) / model_param[2]

# Daten und Trennlinie visualisieren
plt.plot(x1_values, x2_values, color='green', label='Trennlinie')

# Aufgabe 1d
# ##########
# Bringen Sie diesen Code ans Laufen indem Sie ggf. die Variablen w0, w1 und w1 an ihr Skript anpassen.
#
x1, x2 = np.meshgrid(np.linspace(-11,11,10), np.linspace(-11,11,10))
z = model_param[0] + model_param[1] * x1 + model_param[2] * x2**2
plt.contourf(x1, x2, z, levels=[-10,0,10], colors=["b","r"], alpha=.2)
plt.contour(x1, x2, z, levels=[0], colors=["k"])
plt.xlim((0,7))
plt.ylim((0,6))
plt.show()