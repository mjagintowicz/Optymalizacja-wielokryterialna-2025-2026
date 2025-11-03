import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from problem_definition import objective_F1, objective_F2


center = np.array([3.0, 4.0])
radius = 1.0

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

bounds = [(center[0]-radius, center[0]+radius), (center[1]-radius, center[1]+radius)]
constraints = {'type': 'ineq', 'fun': inside_circle}
x0 = center.copy()

# Generowanie punktów Pareto
pareto_data = []
weights = np.linspace(0, 1, 50)
for w1 in weights:
    w2 = 1 - w1
    res = minimize(linear_scalarization, x0, args=(w1, w2), method='SLSQP',
                   bounds=bounds, constraints=constraints)
    if res.success:
        x_opt, y_opt = res.x
        pareto_data.append([F1(x_opt, y_opt), F2(x_opt, y_opt), x_opt, y_opt])

pareto_data = np.array(pareto_data)

# Generowanie punktów okręgu do rysunku
theta = np.linspace(0, 2*np.pi, 200)
circle_x = center[0] + radius * np.cos(theta)
circle_y = center[1] + radius * np.sin(theta)

# Przekształcenie okręgu przez funkcje celu
transformed_x = F1(circle_x, circle_y)
transformed_y = F2(circle_x, circle_y)

# Wykresy
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 1. Oryginalny okrąg + punkty Pareto w przestrzeni decyzyjnej
axs[0].plot(circle_x, circle_y, label='Okrąg', color='b')
axs[0].scatter(pareto_data[:,2], pareto_data[:,3], color='g', label='Punkty Pareto')
axs[0].set_aspect('equal')
axs[0].set_title("Okrąg w przestrzeni decyzyjnej")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()
axs[0].grid(True)

# 2. Przekształcony okrąg + punkty Pareto w przestrzeni funkcji celu
axs[1].plot(transformed_x, transformed_y, label='Przekształcony okrąg', color='g')
axs[1].scatter(pareto_data[:,0], pareto_data[:,1], color='m', label='Punkty Pareto')
axs[1].set_aspect('equal')
axs[1].set_title("Przekształcony okrąg w przestrzeni F1, F2")
axs[1].set_xlabel("F1")
axs[1].set_ylabel("F2")
axs[1].legend()
axs[1].grid(True)

plt.suptitle("Skalaryzacja liniowa z wagami")
plt.tight_layout()
plt.show()