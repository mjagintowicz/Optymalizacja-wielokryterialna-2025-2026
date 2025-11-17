import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from problem_definition import objective_G1, objective_G2, objective_G3, space_properties_3d


# ======== Funkcje celu ========
G1 = objective_G1()
G2 = objective_G2()
G3 = objective_G3()

# ======== Przestrzeń decyzyjna — SFERA ========
x_center, y_center, z_center, r = space_properties_3d()
center = np.array([x_center, y_center, z_center])
x0 = center.copy()


# --- Funkcja skalaryzacji liniowej ---
def linear_scalarization(vars, w1, w2, w3):
    x, y, z = vars
    return w1*G1(x, y, z) + w2*G2(x, y, z) + w3*G3(x, y, z)


# --- Ograniczenie: wewnątrz sfery ---
def inside_sphere(vars):
    x, y, z = vars
    return r**2 - ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

def compute_pareto_weights_3d(n_weights=50):

    bounds = [(x_center-r, x_center+r), (y_center-r, y_center+r), (z_center-r, z_center+r)]
    constraints = {'type': 'ineq', 'fun': lambda v: inside_sphere}

    pareto_dec = []
    pareto_obj = []

    weights = np.linspace(0, 1, n_weights)
    for w1 in weights:
        for w2 in weights:
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue

            res = minimize(linear_scalarization, x0, args=(w1, w2, w3, G1, G2, G3), method="SLSQP", bounds=bounds,
                constraints=constraints)

            if res.success:
                x_opt, y_opt, z_opt = res.x
                pareto_dec.append([x_opt, y_opt, z_opt])
                pareto_obj.append([G1(x_opt, y_opt, z_opt), G2(x_opt, y_opt, z_opt), G3(x_opt, y_opt, z_opt)])

    pareto_dec = np.array(pareto_dec)
    pareto_obj = np.array(pareto_obj)

    # --- Siatka sfery ---
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    sphere_x = x_center + r * np.sin(theta) * np.cos(phi)
    sphere_y = y_center + r * np.sin(theta) * np.sin(phi)
    sphere_z = z_center + r * np.cos(theta)

    # --- Przekształcenie sfery do przestrzeni funkcji celu ---
    sphere_G1 = G1(sphere_x, sphere_y, sphere_z)
    sphere_G2 = G2(sphere_x, sphere_y, sphere_z)
    sphere_G3 = G3(sphere_x, sphere_y, sphere_z)

    return pareto_dec, pareto_obj, sphere_x, sphere_y, sphere_z, sphere_G1, sphere_G2, sphere_G3
