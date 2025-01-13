import numpy as np
from matplotlib import pyplot as plt

# Einleitung
# #########
# Gegeben sind nun diese Daten
coords_x1 = np.random.normal(0, 0.5, 100) 
coords_x2 = np.random.normal(0, 0.5, 100)
class_y = 2.0 * ((coords_x1**2 + coords_x2**2) < 0.4) - 1.0


# Aufgabe 3a
# ##########
# Schätzen Sie ein Modell der Form 
#
#   y = w0 + w1 * x1 + w2 * x2 + w3 * x1^2 + w4 * x2^2 + w5*x1*x2
# 
# und zeichen Sie die Daten sowie die Trennfläche ihres Modells
X = np.stack([np.ones_like(coords_x1), coords_x1, coords_x2, coords_x1**2, coords_x2**2, coords_x1*coords_x2], 1)

pseudoinv = np.linalg.pinv(X)
model_param = pseudoinv @ class_y

print(f"Pseudoinverse : {pseudoinv}")
print("Parameter : ", model_param)

plt.scatter(coords_x1[class_y == -1], coords_x2[class_y == -1], color = "Red")
plt.scatter(coords_x1[class_y == 1], coords_x2[class_y == 1], color = "blue")



## Bringen Sie diesen Code wieder ans Laufen um die Trennfläche zu visualieren
x1, x2 = np.meshgrid(np.linspace(-2,2,50), np.linspace(-2,2,50))
z = model_param[0] + model_param[1] * x1 + model_param[2] * x2 + model_param[3] * (x1 ** 2) + model_param[4] * (x2 ** 2) + model_param[5] * x1 * x2
z = np.clip(z, -1, 1)
print(np.min(z), np.max(z))
blue_indices = (class_y < 0.0)
red_indices = (class_y > 0.0)
plt.contourf(x1, x2, z, levels=np.linspace(-1,1,20), cmap="coolwarm", alpha=.2)
plt.contour(x1, x2, z, levels=[0.0], colors=["k"])
plt.plot(coords_x1[blue_indices], coords_x2[blue_indices], 'bo')
plt.plot(coords_x1[red_indices], coords_x2[red_indices], 'ro')


plt.xlim((-1,1))
plt.ylim((-1,1))
plt.show()