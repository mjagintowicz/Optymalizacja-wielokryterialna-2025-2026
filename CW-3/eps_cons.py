import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from problem_definition import objective_F1, objective_F2, space_properties

f1 = objective_F1()
f2 = objective_F2()
x_center, y_center, r = space_properties()

def objective(x):
    """
    Funkcja celu F1
    """
    return f1(x[0], x[1])

def cons_eps(x, eps):
    """
    Ograniczenie F2 <= eps
    """
    return eps - f2(x[0], x[1])

def cons_circle(x):
    """
    Ograniczenie obszaru
    """
    return r**2 - ((x[0]-x_center)**2 + (x[1]-y_center)**2)

def compute_pareto_epsilon(n=20, eps_min=17, eps_max=30):

    eps_vector = np.linspace(eps_min, eps_max, num=n)

    x0 = np.array([x_center, y_center])
    bounds = [(x_center - r, x_center + r), (y_center - r, y_center + r)]

    pareto_points = []

    for eps in eps_vector:
        constraint_eps = {'type': 'ineq', 'fun': lambda x, e=eps: e - f2(x[0], x[1])}
        constraint_circle = {'type': 'ineq', 'fun': cons_circle}

        sol = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[constraint_eps, constraint_circle],
            options={'maxiter': 100, 'disp': False}
        )

        if sol.success:
            xopt, yopt = sol.x
            pareto_points.append([f1(xopt, yopt), f2(xopt, yopt), xopt, yopt])

    pareto_points = np.array(pareto_points)

    # --- okrąg ---
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = x_center + r * np.cos(theta)
    circle_y = y_center + r * np.sin(theta)

    # --- obraz okręgu ---
    transformed_x = f1(circle_x, circle_y)
    transformed_y = f2(circle_x, circle_y)

    return pareto_points, circle_x, circle_y, transformed_x, transformed_y


def plot_pareto_epsilon(pareto, circle_x, circle_y, tx, ty):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Lewy wykres – przestrzeń decyzyjna
    ax[0].plot(circle_x, circle_y, label="Okrąg dopuszczalny", color='b')
    ax[0].plot(pareto[:, 2], pareto[:, 3], 'ro', label="Punkty Pareto")
    ax[0].set_title("Okrąg i punkty Pareto")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend()
    ax[0].axis('equal')

    # Prawy wykres – obraz w przestrzeni funkcji celu
    ax[1].plot(tx, ty, label="Przekształcony okrąg", color='g')
    ax[1].plot(pareto[:, 0], pareto[:, 1], 'mo', label="Punkty Pareto")
    ax[1].set_title("Obraz okręgu w przestrzeni (F1, F2)")
    ax[1].set_xlabel("F1")
    ax[1].set_ylabel("F2")
    ax[1].legend()
    ax[1].axis('equal')

    plt.suptitle(r"Skalaryzacja metodą $\epsilon$-ograniczeń")
    plt.tight_layout()

    return fig