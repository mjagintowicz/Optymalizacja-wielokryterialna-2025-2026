import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from problem_definition import objective_F1, objective_F2, space_properties

f1 = objective_F1()
f2 = objective_F2()
x_center, y_center, r = space_properties()

n = 20
epsilon_min = 17
epsilon_max = 30
epsilon_vector = np.linspace(epsilon_min, epsilon_max, num=n)

x0 = np.array([x_center, y_center])
bounds = [(x_center-r, x_center+r), (y_center-r, y_center+r)]

pareto_points = []

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


for eps in epsilon_vector:

    constraint_eps = {'type': 'ineq', 'fun': lambda x, e=eps: e - f2(x[0], x[1])}
    constraint_circle = {'type': 'ineq', 'fun': cons_circle}
    constraints = [constraint_eps, constraint_circle]

    sol = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100, 'disp': False}
    )

    if sol.success:
        xopt, yopt = sol.x
        pareto_points.append([f1(xopt, yopt), f2(xopt, yopt), xopt, yopt])

pareto_points = np.array(pareto_points)

theta = np.linspace(0, 2*np.pi, 200)
circle_x = x_center + r * np.cos(theta)
circle_y = y_center + r * np.sin(theta)

transformed_x = f1(circle_x, circle_y)
transformed_y = f2(circle_x, circle_y)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(circle_x, circle_y, label="Okrąg dopuszczalny", color='b')
plt.plot(pareto_points[:, 2], pareto_points[:, 3], 'ro', label="Punkty Pareto")
plt.title("Okrąg i punkty Pareto")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(transformed_x, transformed_y, label="Przekształcony okrąg", color='g')
plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'mo', label="Punkty Pareto")
plt.title("Obraz okręgu w przestrzeni (F1, F2)")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend()
plt.axis('equal')

plt.suptitle(r"Skalaryzacja metodą $\epsilon$-ograniczeń")
plt.tight_layout()
plt.show()