import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from problem_definition import objective_F1, objective_F2, space_properties

# Funkcje celu
F1 = objective_F1()
F2 = objective_F2()

# Ograniczenia: punkt w obrębie koła
x, y, radius = space_properties()
center = np.array([x, y])
x0 = center.copy()

def inside_circle(vars, center, radius):
    x, y = vars
    return radius**2 - ((x - center[0])**2 + (y - center[1])**2)


def distance_scalarization(vars, lambda_vec, x_star, F1, F2):
    x, y = vars
    f_vals = np.array([F1(x, y), F2(x, y)])
    return np.linalg.norm(lambda_vec * (f_vals - x_star), ord=2)


def compute_pareto_ideal_point(n_weights=100, w_min=0.1, w_max=1.0):

    F1 = objective_F1()
    F2 = objective_F2()
    x_c, y_c, radius = space_properties()

    center = np.array([x_c, y_c])
    x0 = center.copy()

    bounds = [(x_c - radius, x_c + radius), (y_c - radius, y_c + radius)]

    # Punkt dominujący F1* i F2*
    res1 = minimize(lambda v: F1(v[0], v[1]), x0, method='SLSQP', bounds=bounds,
        constraints={'type': 'ineq', 'fun': lambda v: inside_circle(v, center, radius)})
    F1_star = F1(res1.x[0], res1.x[1])

    res2 = minimize(lambda v: F2(v[0], v[1]), x0, method='SLSQP', bounds=bounds,
        constraints={'type': 'ineq', 'fun': lambda v: inside_circle(v, center, radius)})
    F2_star = F2(res2.x[0], res2.x[1])

    x_star = np.array([F1_star, F2_star])

    # Skalryzacja przez odległość
    weights = np.linspace(w_min, w_max, n_weights)
    pareto_points = []

    for w in weights:
        lambda_vec = np.array([w, 1 - w])

        result = minimize(
            lambda v: distance_scalarization(v, lambda_vec, x_star, F1, F2), center, method='SLSQP', bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda v: inside_circle(v, center, radius)})

        if result.success:
            x_opt, y_opt = result.x
            pareto_points.append([F1(x_opt, y_opt), F2(x_opt, y_opt), x_opt, y_opt])

    pareto_points = np.array(pareto_points)

    # Okrąg i jego obraz
    theta = np.linspace(0, 2*np.pi, 1000)
    circle_x = x_c + radius * np.cos(theta)
    circle_y = y_c + radius * np.sin(theta)

    transformed_x = F1(circle_x, circle_y)
    transformed_y = F2(circle_x, circle_y)

    return pareto_points, circle_x, circle_y, transformed_x, transformed_y, x_star
