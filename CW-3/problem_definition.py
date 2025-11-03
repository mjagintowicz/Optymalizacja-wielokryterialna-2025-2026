import numpy as np
import matplotlib.pyplot as plt

def objective_F1():
    return lambda x, y: -2 * x**2 + 0.5 * y**2

def objective_F2():
    return lambda x, y: x ** 2 + y ** 2

def space_properties():
    x = 3
    y = 4
    radius = 2
    return x, y, radius

def visualisation():
    x, y, r = space_properties()

    t = np.linspace(0, 2 * np.pi, 1000)
    x_vals = x + r * np.cos(t)
    y_vals = y + r * np.sin(t)

    F1 = objective_F1()
    F2 = objective_F2()

    f1_vals = F1(x_vals, y_vals)
    f2_vals = F2(x_vals, y_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(f1_vals, f2_vals, color='blue')
    plt.title("Koło przekształcone za pomocą nieliniowych F1, F2")
    plt.xlabel("F1(x, y) = x * y^2")
    plt.ylabel("F2(x, y) = x^2 + y^2")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    visualisation()