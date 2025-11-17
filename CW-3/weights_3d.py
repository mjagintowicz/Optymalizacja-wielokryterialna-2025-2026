import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from problem_definition import objective_G1, objective_G2, objective_G3, space_properties_3d

# ======== Przestrzeń decyzyjna — SFERA ========
cx, cy, cz, radius = space_properties_3d()
center = np.array([cx, cy, cz])

# ======== Funkcje celu ========
G1 = objective_G1()
G2 = objective_G2()
G3 = objective_G3()

# ======== Funkcja skalaryzowana ========
def linear_scalarization(vars, w1, w2, w3):
    x, y, z = vars
    return w1 * G1(x, y, z) + w2 * G2(x, y, z) + w3 * G3(x, y, z)

# ======== Ograniczenie: wewnątrz sfery ========
def inside_sphere(vars):
    x, y, z = vars
    return radius**2 - ((x - cx)**2 + (y - cy)**2 + (z - cz)**2)

bounds = [
    (cx-radius, cx+radius),
    (cy-radius, cy+radius),
    (cz-radius, cz+radius)
]

constraints = {'type': 'ineq', 'fun': inside_sphere}
x0 = center.copy()

# ======== GENEROWANIE FRONTU PARETO ========
pareto_dec = []   # punkty w przestrzeni decyzyjnej
pareto_obj = []   # punkty w przestrzeni funkcji celu

weights = np.linspace(0, 1, 50)

for w1 in weights:
    for w2 in weights:
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue

        res = minimize(
            linear_scalarization,
            x0,
            args=(w1, w2, w3),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if res.success:
            x_opt, y_opt, z_opt = res.x
            pareto_dec.append([x_opt, y_opt, z_opt])
            pareto_obj.append([G1(x_opt, y_opt, z_opt),
                               G2(x_opt, y_opt, z_opt),
                               G3(x_opt, y_opt, z_opt)])

pareto_dec = np.array(pareto_dec)
pareto_obj = np.array(pareto_obj)

# ======== GENEROWANIE SIATKI SFERY ========
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

sphere_x = cx + radius * np.sin(theta) * np.cos(phi)
sphere_y = cy + radius * np.sin(theta) * np.sin(phi)
sphere_z = cz + radius * np.cos(theta)

# ======== PRZEKSZTAŁCENIE SFERY DO PRZESTRZENI G ========
sphere_G1 = G1(sphere_x, sphere_y, sphere_z)
sphere_G2 = G2(sphere_x, sphere_y, sphere_z)
sphere_G3 = G3(sphere_x, sphere_y, sphere_z)

# ======== RYSOWANIE ========
fig = plt.figure(figsize=(16, 7))

# ---- Wykres 1: Sfera + punkty Pareto w przestrzeni decyzyjnej ----
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.15, color='lightblue')
ax1.scatter(pareto_dec[:,0], pareto_dec[:,1], pareto_dec[:,2],
            c='red', s=40, label="Punkty Pareto")
ax1.set_title("Sfera w przestrzeni decyzyjnej")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.legend()

# ---- Wykres 2: Przekształcona sfera + front Pareto ----
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(sphere_G1, sphere_G2, sphere_G3, alpha=0.2, color="lightgreen")
ax2.scatter(pareto_obj[:,0], pareto_obj[:,1], pareto_obj[:,2],
            c='purple', s=40, label="Front Pareto")
ax2.set_title("Sfera przekształcona przez G1, G2, G3")
ax2.set_xlabel("G1")
ax2.set_ylabel("G2")
ax2.set_zlabel("G3")
ax2.legend()

plt.suptitle("Skalaryzacja liniowa — wersja 3D", fontsize=16)
plt.tight_layout()
plt.show()
