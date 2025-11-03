import numpy as np
import matplotlib.pyplot as plt

x = 3
y = 4
r = 2

t = np.linspace(0, 2 * np.pi, 1000)
x_vals = x + r * np.cos(t)
y_vals = y + r * np.sin(t)

f1_vals = -2 * x_vals**2 + 0.5 * y_vals**2
f2_vals = x_vals**2 + y_vals**2

plt.figure(figsize=(8, 6))
plt.plot(f1_vals, f2_vals, color='blue')
plt.title("Koło przekształcone za pomocą nieliniowych F1, F2")
plt.xlabel("F1(x, y) = x * y^2")
plt.ylabel("F2(x, y) = x^2 + y^2")
plt.grid()
plt.show()