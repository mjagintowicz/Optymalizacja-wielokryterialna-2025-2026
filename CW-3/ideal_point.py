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

def inside_circle(vars):
    x, y = vars
    return radius**2 - ((x - center[0])**2 + (y - center[1])**2)

bounds = [(center[0]-radius, center[0]+radius), (center[1]-radius, center[1]+radius)]
constraints = {'type': 'ineq', 'fun': inside_circle}

# Punkt dominujący w przestrzeni funkcji celu
res1 = minimize(lambda v: F1(v[0], v[1]), x0, method='SLSQP', bounds=bounds, constraints=constraints)
F1_star = F1(res1.x[0], res1.x[1])

# Minimalizacja F2
res2 = minimize(lambda v: F2(v[0], v[1]), x0, method='SLSQP', bounds=bounds, constraints=constraints)
F2_star = F2(res2.x[0], res2.x[1])

# Punkt idealny
x_star = np.array([F1_star, F2_star])

# Wagi dla skalaryzacji
lambda_vec = np.array([1.0, 1.0])

# Funkcja celu skalaryzowana przez odległość (p=2 -> euklidesowa)
def distance_scalarization(vars):
    x, y = vars
    f_vals = np.array([F1(x, y), F2(x, y)])
    return np.linalg.norm(lambda_vec * (f_vals - x_star), ord=2)  # p=2


pareto_points = []
weights = np.linspace(0.1, 1.0, 100)
for w in weights:
    lambda_vec = np.array([w, 1-w])
    result = minimize(distance_scalarization, center, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        x_opt, y_opt = result.x
        pareto_points.append([F1(x_opt, y_opt), F2(x_opt, y_opt), x_opt, y_opt])

pareto_points = np.array(pareto_points)

# Generowanie granicy koła
theta = np.linspace(0, 2*np.pi, 1000)
circle_x = center[0] + radius * np.cos(theta)
circle_y = center[1] + radius * np.sin(theta)

# Przekształcenie koła przez funkcje celu
transformed_x = F1(circle_x, circle_y)
transformed_y = F2(circle_x, circle_y)

# Punkt najbliższy względem idealnego
distances_to_ideal = np.linalg.norm(pareto_points[:, :2] - x_star, axis=1)
best_index = np.argmin(distances_to_ideal)

best_f1, best_f2, best_x, best_y = pareto_points[best_index]

# Wykresy
plt.figure(figsize=(14, 8))

# Oryginalny okrąg + punkty Pareto
plt.subplot(1,2,1)
plt.fill(circle_x, circle_y, color='lightblue', alpha=0.3, label='Obszar okręgu')
plt.plot(circle_x, circle_y, color='green', linewidth=2, linestyle='--', label='Granica okręgu')
plt.scatter(pareto_points[:,2], pareto_points[:,3], color='blue', s=70, label='Punkty Pareto')
plt.scatter(best_x, best_y, color='red', s=140, edgecolor='black',
            label='Najbliższy punkt do idealnego')
plt.title("Przestrzeń decyzyjna")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.legend()
plt.grid(True, linestyle=':')

# Przekształcony okrąg + punkty Pareto
plt.subplot(1,2,2)
plt.fill(transformed_x, transformed_y, color='lightgreen', alpha=0.25, label='Obszar przekształcony')
plt.plot(transformed_x, transformed_y, color='green', linewidth=2, linestyle='-', label='Granica okręgu')
plt.scatter(pareto_points[:,0], pareto_points[:,1], color='blue', s=70, label='Punkty Pareto')
plt.scatter(best_f1, best_f2, color='red', s=140, edgecolor='black',
            label='Najbliższy punkt do idealnego')
plt.title("Przestrzeń funkcji celu F1-F2")
plt.xlabel("F1")
plt.ylabel("F2")
plt.axis('equal')
plt.legend()
plt.grid(True, linestyle='--')

plt.suptitle("Skalaryzacja przez odległość od punktu dominującego", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()
