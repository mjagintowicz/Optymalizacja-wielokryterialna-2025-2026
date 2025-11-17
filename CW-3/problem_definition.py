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

def space_properties_3d():
    x = 3
    y = 4
    z = 2
    radius = 2
    return x, y, z, radius

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


def objective_G1():
    return lambda x, y, z: x**2 + y**3 + z  # nieliniowe

def objective_G2():
    return lambda x, y, z: 2 * x + y**2 + z  # nieliniowe

def objective_G3():
    return lambda x, y, z: -x - y - z ** 2  # liniowe


def visualisation_three_criteria():
    cx, cy, cz, r = space_properties_3d()

    # parametry sfery
    theta = np.linspace(0, np.pi, 80)
    phi = np.linspace(0, 2*np.pi, 160)
    theta, phi = np.meshgrid(theta, phi)

    # współrzędne powierzchni sfery
    x_vals = cx + r * np.sin(theta) * np.cos(phi)
    y_vals = cy + r * np.sin(theta) * np.sin(phi)
    z_vals = cz + r * np.cos(theta)

    # Funkcje celu przyjmują teraz (x, y, z)
    G1 = objective_G1()
    G2 = objective_G2()
    G3 = objective_G3()

    g1_vals = G1(x_vals, y_vals, z_vals)
    g2_vals = G2(x_vals, y_vals, z_vals)
    g3_vals = G3(x_vals, y_vals, z_vals)

    # === Wizualizacja 3D frontu G1-G2-G3 ===
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    ax.plot_surface(g1_vals, g2_vals, g3_vals, cmap="viridis", alpha=0.85)
    ax.set_title("Sfera przekształcona przez G1, G2, G3 (3D)")
    ax.set_xlabel("G1")
    ax.set_ylabel("G2")
    ax.set_zlabel("G3")

    # === Rzuty 2D ===
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(g1_vals, g2_vals, s=1, color="red", label="Rzut (G1, G2)")
    ax2.scatter(g1_vals, g3_vals, s=1, color="green", label="Rzut (G1, G3)")
    ax2.scatter(g2_vals, g3_vals, s=1, color="blue", label="Rzut (G2, G3)")
    ax2.set_title("Rzuty 2D przekształconej sfery")
    ax2.set_xlabel("wartości funkcji G")
    ax2.set_ylabel("wartości funkcji G")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    visualisation()
    visualisation_three_criteria()