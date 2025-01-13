import numpy as np
from matplotlib import pyplot as plt

# Aufgabe 1a
############

# Erzeugen Sie mit np.linspace ein NumPy Array mit 100 Werten zwischen 0 und 15
# Berechnen Sie dann die dazugehörigen Funktionswerte der Funktion
#   
#   f(x) = exp(sin(x))
#
# Leiten Sie f ab und berechnen Sie ebenfalls die Funktionswerte der Ableitung
#
# Plotten Sie f(x) und f'(x) mit plt.plot(...) und plt.show(...)

# Create a linear space between 0 and 5
x=np.linspace(0.00, 15.00, num=100)
print("X:" , x)
print(x)

f= np.exp(np.sin(x))
plt.plot(x,f)


dx = np.cos(x) * np.exp(np.sin(x))
plt.plot(x, dx)


## Aufgabe 1b
#############

# Approximieren Sie die Ableitung nummerisch über
# 
#   f'(x) ~ dy/dx = (f(x+1) - f(x-1)) / dx
# 
# Überlegen Sie was genau dx in diesem Zusammenhang ist und plotten Sie 
# ihre numerische Approximation ins gleiche Koordinatensystem

# Approximate derivative numerically
dx=(x[1] - x[0])*2
print("dx = ",dx)

df_approx = np.zeros(100)
for i in range(1,99):
    df_approx[i] =(f[i+1] - f[i-1]) /dx

print (df_approx)

plt.plot(x, df_approx)
plt.show()