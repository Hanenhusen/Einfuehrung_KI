import numpy as np
from matplotlib import pyplot as plt
import pickle
import os

os.system("cls")

# Einleitung
############
#
# Sie sind Teil eines internationalen Teams aus Physikern und Datenwissenschaftlern.
# Als ersten Test einer guten Zusammenarbeit soll ihr Team die Gravitationskonstante der Erde 
# bestimmen. Ihre Physikkollegen schlagen dazu einen Frei-Fall-Test vor, d.h. ein Gegenstand 
# wird auf bekannter Höhe (100m) in einer Vakuumröhre fallen gelassen. Die Höhe des Gegenstands
# im freien Fall wird zu verschiedenen Zeitpunkten mittels Radar gemessen und in einer Zeitreihe 
# aufgetragen. Ihr Physikerkollegen haben ihnen die aktuellste Datenmeßreihe geschickt und 
# sie gebeten daraus eine möglichst gute Schätzung abzuleiten. 
#
# Das folgende Skript öffnet die Daten und speicher sie in zwei numpy-Arrays (time und height)
with open("loss/data.pk", "rb") as f:
  time,height = pickle.loads(f.read())

# Aufgabe 1
# #########
# Visualisieren Sie die Daten mittels MatPlotLib um sich einen ersten Überblick zu verschaffen

plt.scatter(time, height)
plt.xlabel("Zeit (s)")
plt.ylabel("Höhe (m)")
# plt.show()

# Aufgabe 2
# #########
# Offensichtlich sind die Daten mit Ausreißern kontaminiert. Versuchen Sie dennoch zunächst eine
# Schätzung mit Hilfe eines linearen Regressionsmodells. 
# Hinweis: Ihr Modell lautet 
#   
#     h = 100m - a * t²
#
# und hat somit nur einen Parameter (a). Überlegen Sie wie Sie die bekannte Ausgangshöhe (100m)
# einbauen können 

constante_hoehe = 100.0

X = -time**2

a = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ (height-constante_hoehe)

print(f"Schätzung für a: {a}")

# Aufgabe 3
# #########
# Implementieren Sie nun 500 Schritte eines Gradientenabstiegsverfahren um ihre Schätzung zu verbessern.
# Verwenden Sie dazu den MAE-Loss wie in der Vorlesung diskutiert. Starten Sie mit der (absichtlich falschen) 
# Annahme a=0.0m/s² und einer Lernrate von 0.005. 
# Geben Sie in jedem Schritt die aktuelle Schätzung (a), den MAE sowie den Gradienten aus.
#
# Was fällt ihnen gegen Ende der 500 Schritte auf?
#
# Bonus: Zeichnen Sie in jedem 10.ten Schritt die Daten sowie die von ihrem Modell geschätzte Fall-Kurve. Verwenden
# Sie plt.pause(0.01) um die Darstellung zu ermöglichen ohne auf einen Tastendruck warten zu müssen.

def lineaers_modell(a, t):
  return constante_hoehe - a * t**2

def residuum(a, time, height):
  return lineaers_modell(a, time) - height

# print(f"Residuum: {residuum(a, time, height)}")

schritte = 500
eta = 0.005
a = 0.0


for i in range(schritte):
  gradient_loss = np.sum(np.sign(residuum(a, time[:,0], height)) * -time[:,0]**2)
  a = a - eta * gradient_loss
  mae_loss = np.sum(np.abs(residuum(a, time[:,0], height))) / len(time[:,0])
  print(f"Schätzung für a: {a}; MAE: {mae_loss}; Gradient: {gradient_loss}")

  if i % 10 == 0:
    plt.title("MAE-Loss")
    plt.scatter(time, height)
    plt.plot(time, lineaers_modell(a, time))
    plt.pause(0.01)
    plt.clf()

# Aufgabe 4
# #########
# Implementieren Sie nun wieder 500 Schritte eines Gradientenabstiegsverfahren um ihre Schätzung zu verbessern.
# Verwenden Sie dazu aber der LogCosh-Loss wie in der Vorlesung diskutiert. Starten Sie wieder mit der 
# (absichtlich falschen) Annahme a=0.0m/s² und einer Lernrate von 0.005. 
# Geben Sie in jedem Schritt wieder die aktuelle Schätzung (a), den LogCosh-Loss sowie den Gradienten aus.
#
# Bonus: Zeichnen Sie in jedem 10.ten Schritt die Daten sowie die von ihrem Modell geschätzte Fall-Kurve. Verwenden
# Sie plt.pause(0.01) um die Darstellung zu ermöglichen ohne auf einen Tastendruck warten zu müssen.

def logcosh_loss(a, time, height):
  return np.sum(np.log(np.cosh(residuum(a, time, height)))) / len(time)

a = 0.0


for i in range(schritte):
  gradient_loss = np.sum(np.tanh(residuum(a, time[:,0], height)) * -time[:,0]**2)
  
  a = a - eta * gradient_loss
  
  logcosh = logcosh_loss(a, time[:,0], height)
  
  print(f"Schätzung für a: {a}; LogCosh: {logcosh}; Gradient: {gradient_loss:.15f}")

  if i % 10 == 0:
    plt.title("LogCosh-Loss")
    plt.scatter(time, height)
    plt.plot(time, lineaers_modell(a, time))
    plt.pause(0.01)
    plt.clf()

plt.close()

# Aufgabe 5
# #########
# Berechnen Sie für a=9.81m/² und jeden Datenpunkt das Residuum r sowie den hyperbolischen Tangens von r.
# Berechnen Sie dann den Quotienten des hyperbolischen Tangens mit dem tatsächlichen Residuum.
# Plotten Sie die Daten farbig, wobei sie den Betrag des Quotienten für die Farbkodierung verwenden. Interpretieren
# sie das Ergebniss.
#
# Hinweis: Verwenden Sie plt.scatter mit der "coolwarm"-Colormap (cmap Parameter)
#
#   https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html


a = 9.81

r = residuum(a, time[:,0], height)

tanh_r = np.tanh(r)

quotient = np.abs(tanh_r) / np.abs(r)

plt.scatter(time, height, c=quotient, cmap="coolwarm")
plt.xlabel("Zeit (s)")
plt.ylabel("Höhe (m)")
plt.colorbar()
plt.show()


  
    


