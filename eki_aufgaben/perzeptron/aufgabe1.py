import math
import numpy as np
from matplotlib import pyplot as plt

# In dieser Übung sollen sie den Perzeptron-Lernalgorithmus aus der Vorlesung implementieren.
# Dazu betrachten wir zunächst die beiden Mengen P und N 
P = np.array([[1.0,1.0,1.0],
              [2.0,2.0,1.0],
              [4.0,3.0,1.0],
              [1.0,3.5,1.0],
              [1.5,2.5,1.0]
              ])

N = np.array([[3.0,1.0,1.0],
              [4.0,2.0,1.0],
              [5.0,2.0,1.0],
              [4.0,1.0,1.0],
              [3.0,1.5,1.0]
              ])

# Unser Perzeptron verwendet die Entscheidungsfunktion 
#
#                    w * x >= 0     (1)
#   w1 * x1 + w2 * x2 + w3 >= 0     (2)
#
# wobei x ein Vektor aus P oder N ist und w = (w1, w2, w3) der dreidimensionale 
# Parametervektor des Perzeptrons.

# Aufgabe 1
# Bestimmen Sie ausgehend von dem Gewichtsvektor w = (-0.5, 1.0, 0.5) für jeden Punkt aus P und N, wie
# dieser von Perzeptron klassifiziert werden würde

w = np.array([-0.5, 1.0, 0.5])
print("-----------Aufgabe 1-----------")
print(P @ w > 0)
print(N @ w < 0)



# Aufgabe 2
# Implementieren Sie eine Funktion "plot_data" welche die beiden Mengen P und N sowie
# die durch w bestimmte Trennfläche plottet. Stellen Sie dazu Gleichung (2) (s. oben)
# nach y um. 

# Funktion zum Plotten der Daten und der Trennfläche
def plot_data(P, N, w):
    plt.figure(figsize=(10, 6))

    # Plotten der Punkte aus P (positiv, rot) und N (negativ, blau)
    plt.scatter(P[:, 0], P[:, 1], color='blue', marker='o', label='P (positive)')
    plt.scatter(N[:, 0], N[:, 1], color='red', marker='x', label='N (negative)')

    # Bestimmen der Trennfläche
    x1_vals = np.linspace(0, 6, 100)
    x2_vals = -w[2]/ w[1]  - w[0]/ w[1] * x1_vals

    # Plotten der Trennfläche
    plt.plot(x1_vals, x2_vals, "k")

    # Achsen und Titel
    plt.xlabel('x1')
    plt.ylabel('x2')


    plt.show()


# Plotten der Daten und der Trennfläche
plot_data(P, N, w)

# Aufgabe 3
# Konkatenieren Sie nun wie in der Vorlesung die Menge P mit der negierte Menge N um, wie in der Vorlesung, eine Menge 
# B zu erhalten in der alle Datenpunkte x € B auf der positiven Seite der Trennebene liegen müssen.
# Wie überprüfen Sie ob alle Punkte gleichzeitig korrekt klassifiziert werden?



B = np.concatenate((P, -N), 0)

print("-----------Aufgabe 3-----------")
print((B @ w > 0))



# Aufgabe 4
# Implementieren Sie nun den Perzeptron-Lernalgorithmus wie in der Vorlesung gezeigt.
# Starten Sie mit einer While-Schleife, die solange ausgeführt wie nicht alle Punkte
# korrekt klassifiziert werden. Wählen sie mit np.random.uniform einen zufälligen
# Datenpunkt aus B. Überprüfen Sie ob dieser falsch klassifiziert wird (x) und passen sie ggf. 
# den Gewichtsvektor gemäß Vorlesung an. Nutzen Sie die plot_data Methode von oben um den neuen 
# Zustand zu visualisieren. 
#
# Hinweis: Sie müssen mit plt.ion() den interaktiven Modus von matplotlib aktivieren. 
# Mit plt.clf() können sie die aktuelle Figure löschen (zurücksetzen) und mit plt.pause(1) können sie
# eine Sekunde warten (und in dieser Zeit die Grafik anzeigen) bevor das Programm weiter macht.
plt.ion()
while True: # HINWEIS: Fügen Sie z.B. hier ihre Abbruchbedingung ein

    # Pick a random data point
    x = B[np.random.randint(0,len(B))]
    # Check if it is correctly classified, if so skip
    skalar = x @ w
    if (skalar) < 0:
        w = w + x
    plot_data(P, N, w)
    if(B @ w > 0).all():
        break
    else:
        plt.clf()
    # If the data point is not classified correctly,
    # update w vector


    # Draw everything
    # Note: Call plt.clf and plt.pause accordingly


plt.pause(5.0)    