import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from problem_definition import objective_G1, objective_G2, objective_G3, space_properties_3d

# ======== Funkcje celu ========
G1 = objective_G1()
G2 = objective_G2()
G3 = objective_G3()

# ======== Przestrzeń decyzyjna — SFERA ========
x_center, y_center, z_center, r = space_properties_3d()
center = np.array([x_center, y_center, z_center])
x0 = center.copy()

# ======== Funkcja celu ========
def objective(x):
    return G1(x[0], x[1], x[2])

# ======== Ograniczenie: punkt wewnątrz sfery ========
def cons_sphere(x):
    return r**2 - ((x[0]-x_center)**2 + (x[1]-y_center)**2 + (x[2]-z_center)**2)

def compute_pareto_epsilon_3d(n=20, eps_min=20, eps_max=40):

    epsilon_vec = np.linspace(eps_min, eps_max, n)

    bounds = [(x_center - r, x_center + r), (y_center - r, y_center + r), (z_center - r, z_center + r)]

    pareto_dec = []   # punkty decyzyjne
    pareto_obj = []   # punkty w przestrzeni G

    for eps in epsilon_vec:

        cons_eps_G2 = {'type': 'ineq', 'fun': lambda x, e=eps: e - G2(x[0], x[1], x[2])}
        cons_eps_G3 = {'type': 'ineq', 'fun': lambda x, e=eps: e - G3(x[0], x[1], x[2])}
        cons_ball = {'type': 'ineq', 'fun': cons_sphere}

        constraints = [cons_eps_G2, cons_eps_G3, cons_ball]

        sol = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'disp': False})

        if sol.success:
            x_opt, y_opt, z_opt = sol.x

            pareto_dec.append([x_opt, y_opt, z_opt])
            pareto_obj.append([
                G1(x_opt, y_opt, z_opt),
                G2(x_opt, y_opt, z_opt),
                G3(x_opt, y_opt, z_opt)
            ])

    pareto_dec = np.array(pareto_dec)
    pareto_obj = np.array(pareto_obj)

    # ======== SIATKA SFERY ========
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    sphere_x = x_center + r * np.sin(theta) * np.cos(phi)
    sphere_y = y_center + r * np.sin(theta) * np.sin(phi)
    sphere_z = z_center + r * np.cos(theta)

    # ======== PRZEKSZTAŁCONA SFERA ========
    sphere_G1 = G1(sphere_x, sphere_y, sphere_z)
    sphere_G2 = G2(sphere_x, sphere_y, sphere_z)
    sphere_G3 = G3(sphere_x, sphere_y, sphere_z)

    return pareto_dec, pareto_obj, sphere_x, sphere_y, sphere_z, sphere_G1, sphere_G2, sphere_G3
