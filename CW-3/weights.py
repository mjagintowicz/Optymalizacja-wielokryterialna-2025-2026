import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from problem_definition import objective_F1, objective_F2, space_properties

x, y, radius = space_properties()
center = np.array([x, y])

# Funkcje celu
F1 = objective_F1()
F2 = objective_F2()

# Funkcja celu do minimalizacji liniowej kombinacji
def linear_scalarization(vars, w1, w2):
    x, y = vars
    return w1 * F1(x, y) + w2 * F2(x, y)

# Ograniczenia: punkt w obrębie okręgu
def inside_circle(vars):
    x, y = vars
    return radius**2 - ((x - center[0])**2 + (y - center[1])**2)

def compute_pareto_points_weights(n_weights=100):


    bounds = [(center[0] - radius, center[0] + radius),
              (center[1] - radius, center[1] + radius)]
    constraints = {'type': 'ineq', 'fun': inside_circle}
    x0 = center.copy()

    # --- Generowanie danych Pareto ---
    pareto_data = []
    weights = np.linspace(0, 1, n_weights)

    for w1 in weights:
        w2 = 1 - w1
        res = minimize(
            linear_scalarization, x0, args=(w1, w2),
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        if res.success:
            x_opt, y_opt = res.x
            pareto_data.append([F1(x_opt, y_opt), F2(x_opt, y_opt), x_opt, y_opt])

    pareto_data = np.array(pareto_data)

    # Okrąg w przestrzeni decyzyjnej
    theta = np.linspace(0, 2 * np.pi, 1000)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)

    transformed_x = F1(circle_x, circle_y)
    transformed_y = F2(circle_x, circle_y)

    return pareto_data, circle_x, circle_y, transformed_x, transformed_y
