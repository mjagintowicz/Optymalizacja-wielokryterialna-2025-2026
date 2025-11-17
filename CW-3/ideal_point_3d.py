import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from problem_definition import objective_G1, objective_G2, objective_G3, space_properties_3d

# === przygotowanie problemu ===
G1 = objective_G1()
G2 = objective_G2()
G3 = objective_G3()

x_center, y_center, z_center, r = space_properties_3d()
center = np.array([x_center, y_center, z_center])
x0 = center.copy()

# ograniczenie: wewnątrz sfery
def inside_sphere(vars):
    x, y, z = vars
    return r**2 - ((x - x_center) ** 2 + (y - y_center) ** 2 + (z - z_center) ** 2)

# === 2) skalaryzacja przez odległość (p=2) ===
def distance_scalarization(vars, lamb, ideal_point):
    x, y, z = vars
    fvec = np.array([G1(x, y, z), G2(x, y, z), G3(x, y, z)])
    return np.linalg.norm(lamb * (fvec - ideal_point), ord=2)

def compute_pareto_ideal_point_3d(n_weights=100):

    bounds = [
        (x_center - r, x_center + r),
        (y_center - r, y_center + r),
        (z_center - r, z_center + r),
    ]
    constraints = {'type': 'ineq', 'fun': inside_sphere}


    # === 1) znajdowanie punktu idealnego (min każdego Gi osobno) ===
    res_g1 = minimize(lambda v: G1(v[0], v[1], v[2]), x0, method='SLSQP', bounds=bounds, constraints=constraints)
    res_g2 = minimize(lambda v: G2(v[0], v[1], v[2]), x0, method='SLSQP', bounds=bounds, constraints=constraints)
    res_g3 = minimize(lambda v: G3(v[0], v[1], v[2]), x0, method='SLSQP', bounds=bounds, constraints=constraints)

    g1_star = G1(*res_g1.x) if res_g1.success else G1(*x0)
    g2_star = G2(*res_g2.x) if res_g2.success else G2(*x0)
    g3_star = G3(*res_g3.x) if res_g3.success else G3(*x0)

    ideal_point = np.array([g1_star, g2_star, g3_star])

    weights = []
    for i in range(n_weights+1):
        for j in range(n_weights+1-i):
            k = n_weights - i - j
            w1 = i / n_weights
            w2 = j / n_weights
            w3 = k / n_weights
            if (w1 + w2 + w3) > 0:
                weights.append(np.array([w1, w2, w3]))
    weights = np.array(weights)

    pareto_dec = []  # punkty w przestrzeni decyzyjnej (x,y,z)
    pareto_obj = []  # punkty w przestrzeni celów (G1,G2,G3)

    for lamb in weights:
        lamb = np.maximum(lamb, 1e-8)
        res = minimize(distance_scalarization, x0, args=(lamb,ideal_point), method='SLSQP',
                       bounds=bounds, constraints=constraints, options={'maxiter':200})
        if res.success:
            x_opt, y_opt, z_opt = res.x
            pareto_dec.append([x_opt, y_opt, z_opt])
            pareto_obj.append([G1(x_opt, y_opt, z_opt),
                               G2(x_opt, y_opt, z_opt),
                               G3(x_opt, y_opt, z_opt)])

    pareto_dec = np.array(pareto_dec)
    pareto_obj = np.array(pareto_obj)

    theta = np.linspace(0, np.pi, 80)
    phi = np.linspace(0, 2*np.pi, 140)
    theta, phi = np.meshgrid(theta, phi)
    sx = x_center + r * np.sin(theta) * np.cos(phi)
    sy = y_center + r * np.sin(theta) * np.sin(phi)
    sz = z_center + r * np.cos(theta)

    sG1 = G1(sx, sy, sz)
    sG2 = G2(sx, sy, sz)
    sG3 = G3(sx, sy, sz)

    return pareto_dec, pareto_obj, sx, sy, sz, sG1, sG2, sG3, ideal_point, res_g1, res_g2, res_g3

