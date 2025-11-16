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

bounds = [(center[0]-radius, center[0]+radius), (center[1]-radius, center[1]+radius)]
constraints = {'type': 'ineq', 'fun': inside_circle}
x0 = center.copy()

# Generowanie punktów Pareto
pareto_data = []
weights = np.linspace(0, 1, 100)
for w1 in weights:
    w2 = 1 - w1
    res = minimize(linear_scalarization, x0, args=(w1, w2), method='SLSQP',
                   bounds=bounds, constraints=constraints)
    if res.success:
        x_opt, y_opt = res.x
        pareto_data.append([F1(x_opt, y_opt), F2(x_opt, y_opt), x_opt, y_opt])

pareto_data = np.array(pareto_data)

# Generowanie punktów okręgu do rysunku
theta = np.linspace(0, 2*np.pi, 1000)
circle_x = center[0] + radius * np.cos(theta)
circle_y = center[1] + radius * np.sin(theta)

# Przekształcenie okręgu przez funkcje celu
transformed_x = F1(circle_x, circle_y)
transformed_y = F2(circle_x, circle_y)

plt.figure(figsize=(14,8))

# --- Wykres 1: Oryginalny okrąg + punkty Pareto ---
plt.subplot(1,2,1)
plt.fill(circle_x, circle_y, color='lightgreen', alpha=0.3, label='Obszar okręgu')
plt.plot(circle_x, circle_y, color='green', linewidth=2, linestyle='--', label='Granica okręgu')
plt.scatter(pareto_data[:,2], pareto_data[:,3], color='blue', s=70, label='Punkty Pareto')
plt.title("Decyzyjna przestrzeń z punktami Pareto", fontsize=14, fontweight='bold')
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.axis('equal')
plt.legend()
plt.grid(True, linestyle=':', linewidth=1)

# --- Wykres 2: Przekształcony okrąg w przestrzeni F1, F2 ---
plt.subplot(1,2,2)
plt.fill(transformed_x, transformed_y, color='lightgreen', alpha=0.25, label='Obszar przekształcony')
plt.plot(transformed_x, transformed_y, color='green', linewidth=2, linestyle='-', label='Przekształcony okrąg')
plt.scatter(pareto_data[:,0], pareto_data[:,1], color='blue', s=70, label='Punkty Pareto')
plt.title("Przestrzeń funkcji celu F1-F2", fontsize=14, fontweight='bold')
plt.xlabel("F1", fontsize=12)
plt.ylabel("F2", fontsize=12)
plt.axis('equal')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.8)

plt.suptitle("Skalaryzacja liniowa", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()