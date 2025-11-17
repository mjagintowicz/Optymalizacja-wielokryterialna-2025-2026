import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from problem_definition import objective_G1, objective_G2, objective_G3, space_properties_3d

# === przygotowanie problemu ===
G1 = objective_G1()
G2 = objective_G2()
G3 = objective_G3()

cx, cy, cz, r = space_properties_3d()
center = np.array([cx, cy, cz])

# ograniczenie: wewnątrz sfery
def inside_sphere(vars):
    x, y, z = vars
    return r**2 - ((x - cx)**2 + (y - cy)**2 + (z - cz)**2)

bounds = [
    (cx - r, cx + r),
    (cy - r, cy + r),
    (cz - r, cz + r),
]
constraints = {'type': 'ineq', 'fun': inside_sphere}
x0 = center.copy()

# === 1) znajdowanie punktu idealnego (min każdego Gi osobno) ===
res_g1 = minimize(lambda v: G1(v[0], v[1], v[2]), x0, method='SLSQP', bounds=bounds, constraints=constraints)
res_g2 = minimize(lambda v: G2(v[0], v[1], v[2]), x0, method='SLSQP', bounds=bounds, constraints=constraints)
res_g3 = minimize(lambda v: G3(v[0], v[1], v[2]), x0, method='SLSQP', bounds=bounds, constraints=constraints)

g1_star = G1(*res_g1.x) if res_g1.success else G1(*x0)
g2_star = G2(*res_g2.x) if res_g2.success else G2(*x0)
g3_star = G3(*res_g3.x) if res_g3.success else G3(*x0)

ideal_point = np.array([g1_star, g2_star, g3_star])
print("Punkt idealny (G1*,G2*,G3*):", ideal_point)

# === 2) skalaryzacja przez odległość (p=2) ===
def distance_scalarization(vars, lamb):
    x, y, z = vars
    fvec = np.array([G1(x, y, z), G2(x, y, z), G3(x, y, z)])
    return np.linalg.norm(lamb * (fvec - ideal_point), ord=2)

weights = []
n = 16
for i in range(n+1):
    for j in range(n+1-i):
        k = n - i - j
        w1 = i / n
        w2 = j / n
        w3 = k / n
        if (w1 + w2 + w3) > 0:
            weights.append(np.array([w1, w2, w3]))
weights = np.array(weights)

pareto_dec = []  # punkty w przestrzeni decyzyjnej (x,y,z)
pareto_obj = []  # punkty w przestrzeni celów (G1,G2,G3)

for lamb in weights:
    lamb = np.maximum(lamb, 1e-8)
    res = minimize(distance_scalarization, x0, args=(lamb,), method='SLSQP',
                   bounds=bounds, constraints=constraints, options={'maxiter':200})
    if res.success:
        x_opt, y_opt, z_opt = res.x
        pareto_dec.append([x_opt, y_opt, z_opt])
        pareto_obj.append([G1(x_opt, y_opt, z_opt),
                           G2(x_opt, y_opt, z_opt),
                           G3(x_opt, y_opt, z_opt)])

pareto_dec = np.array(pareto_dec)
pareto_obj = np.array(pareto_obj)

if pareto_obj.size == 0:
    raise RuntimeError("Nie znaleziono żadnego punktu Pareto — spróbuj zmienić siatkę wag lub parametry optymalizatora.")

# === 3) wybierz punkt Pareto najbliższy punktowi idealnemu (w przestrzeni celów) ===
dists = np.linalg.norm(pareto_obj - ideal_point, axis=1)
best_idx = np.argmin(dists)
best_obj = pareto_obj[best_idx]
best_dec = pareto_dec[best_idx]
print("Najbliższy punkt Pareto (G):", best_obj, "    (x,y,z):", best_dec)

# === 4) przekształcenie całej sfery do przestrzeni celów ===
theta = np.linspace(0, np.pi, 80)
phi = np.linspace(0, 2*np.pi, 140)
theta, phi = np.meshgrid(theta, phi)
sx = cx + r * np.sin(theta) * np.cos(phi)
sy = cy + r * np.sin(theta) * np.sin(phi)
sz = cz + r * np.cos(theta)

sG1 = G1(sx, sy, sz)
sG2 = G2(sx, sy, sz)
sG3 = G3(sx, sy, sz)

fig = plt.figure(figsize=(16, 7))

# (A) przestrzeń decyzyjna: sfera + punkty Pareto + najlepszy punkt
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(sx, sy, sz, color='lightblue', alpha=0.2, rstride=2, cstride=2, linewidth=0)
if pareto_dec.size:
    ax1.scatter(pareto_dec[:,0], pareto_dec[:,1], pareto_dec[:,2], s=40, c='purple', label='Punkty Pareto')
ax1.scatter([best_dec[0]], [best_dec[1]], [best_dec[2]], s=140, c='red', edgecolor='k',
            label='Najbliższy idealnemu')
# punkty odpowiadające minimom G1, G2, G3
ax1.scatter([res_g1.x[0]], [res_g1.x[1]], [res_g1.x[2]],
            s=120, c='orange', edgecolor='black', marker='X', label='min G1')

ax1.scatter([res_g2.x[0]], [res_g2.x[1]], [res_g2.x[2]],
            s=120, c='green', edgecolor='black', marker='X', label='min G2')

ax1.scatter([res_g3.x[0]], [res_g3.x[1]], [res_g3.x[2]],
            s=120, c='cyan', edgecolor='black', marker='X', label='min G3')
ax1.set_title("Sfera (przestrzeń decyzyjna)")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
ax1.legend()
ax1.view_init(elev=25, azim=45)

# (B) przestrzeń celów: przekształcona sfera + front Pareto + najlepszy punkt
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(sG1, sG2, sG3, cmap='viridis', alpha=0.25, rstride=2, cstride=2, linewidth=0)
if pareto_obj.size:
    ax2.scatter(pareto_obj[:,0], pareto_obj[:,1], pareto_obj[:,2], s=40, c='blue', label='Front Pareto')
ax2.scatter([best_obj[0]], [best_obj[1]], [best_obj[2]], s=140, c='red', edgecolor='k', label='Najbliższy idealnemu')
ax2.scatter(
    [ideal_point[0]], [ideal_point[1]], [ideal_point[2]],
    s=200, c='yellow', edgecolor='black', marker='*', label='Punkt idealny'
)
ax2.set_title("Przekształcona sfera w przestrzeni celów (G1,G2,G3)")
ax2.set_xlabel("G1"); ax2.set_ylabel("G2"); ax2.set_zlabel("G3")
ax2.legend()
ax2.view_init(elev=25, azim=45)

plt.suptitle("Skalaryzacja przez odległość w 3 kryteriach (p=2)", fontsize=16)
plt.tight_layout()
plt.show()
