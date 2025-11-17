import matplotlib.pyplot as plt
import numpy as np


def plot_pareto(pareto_data, circle_x, circle_y, transformed_x, transformed_y, x_star=None):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Wykres przestrzeni decyzyjnej
    ax[0].fill(circle_x, circle_y, color='lightgreen', alpha=0.3, label='Obszar okręgu')
    ax[0].plot(circle_x, circle_y, color='green', linewidth=2, linestyle='--', label='Granica okręgu')
    ax[0].scatter(pareto_data[:, 2], pareto_data[:, 3], color='blue', s=70, label='Punkty Pareto')
    if x_star is not None:
        distances_to_ideal = np.linalg.norm(pareto_data[:, :2] - x_star, axis=1)
        best_index = np.argmin(distances_to_ideal)
        best_f1, best_f2, best_x, best_y = pareto_data[best_index]
        ax[0].scatter(best_x, best_y, color='red', s=140, edgecolor='black', label='Najbliższy punkt do idealnego')

    ax[0].set_title("Decyzyjna przestrzeń z punktami Pareto")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].axis('equal')
    ax[0].grid(True, linestyle=':')

    # Wykres przestrzeni funkcji celu
    ax[1].fill(transformed_x, transformed_y, color='lightgreen', alpha=0.25, label='Obszar przekształcony')
    ax[1].plot(transformed_x, transformed_y, color='green', linewidth=2, label='Granica obszaru')
    ax[1].scatter(pareto_data[:, 0], pareto_data[:, 1], color='blue', s=70, label='Punkty Pareto')
    if x_star is not None:
        ax[1].scatter(best_f1, best_f2, color='red', s=140, edgecolor='black', label='Najbliższy punkt do idealnego')
    ax[1].set_title("Przestrzeń funkcji celu F1-F2")
    ax[1].set_xlabel("F1")
    ax[1].set_ylabel("F2")
    ax[1].axis('equal')
    ax[1].grid(True, linestyle='--')

    plt.legend()
    plt.tight_layout()
    return fig


def plot_pareto_3d(pareto_dec, pareto_obj, sphere_x, sphere_y, sphere_z, sphere_G1, sphere_G2, sphere_G3,
                   ideal_point=None, res_g1=None, res_g2=None, res_g3=None):
    fig = plt.figure(figsize=(16, 7))

    # --- Wykres 1: sfera + punkty Pareto w przestrzeni decyzyjnej ---
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.15, color='lightblue')
    ax1.scatter(pareto_dec[:,0], pareto_dec[:,1], pareto_dec[:,2], c='red', s=40, label="Punkty Pareto")
    if ideal_point is not None:
        dists = np.linalg.norm(pareto_obj - ideal_point, axis=1)
        best_idx = np.argmin(dists)
        best_dec = pareto_dec[best_idx]

        ax1.scatter([best_dec[0]], [best_dec[1]], [best_dec[2]], s=140, c='red', edgecolor='k',
                    label='Najbliższy idealnemu')
        ax1.scatter([best_dec[0]], [best_dec[1]], [best_dec[2]], s=140, c='red', edgecolor='k',
                    label='Najbliższy idealnemu')
        # punkty odpowiadające minimom G1, G2, G3
        ax1.scatter([res_g1.x[0]], [res_g1.x[1]], [res_g1.x[2]],
                    s=120, c='orange', edgecolor='black', marker='X', label='min G1')

        ax1.scatter([res_g2.x[0]], [res_g2.x[1]], [res_g2.x[2]],
                    s=120, c='green', edgecolor='black', marker='X', label='min G2')

        ax1.scatter([res_g3.x[0]], [res_g3.x[1]], [res_g3.x[2]],
                    s=120, c='cyan', edgecolor='black', marker='X', label='min G3')


    ax1.set_title("Sfera w przestrzeni decyzyjnej")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend()

    # --- Wykres 2: przekształcona sfera + front Pareto ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(sphere_G1, sphere_G2, sphere_G3, alpha=0.2, color="lightgreen")
    ax2.scatter(pareto_obj[:,0], pareto_obj[:,1], pareto_obj[:,2], c='purple', s=40, label="Front Pareto")
    if ideal_point is not None:
        best_obj = pareto_obj[best_idx]
        ax2.scatter(pareto_obj[:, 0], pareto_obj[:, 1], pareto_obj[:, 2], s=40, c='blue', label='Front Pareto')
        ax2.scatter([best_obj[0]], [best_obj[1]], [best_obj[2]], s=140, c='red', edgecolor='k',
                    label='Najbliższy idealnemu')
        ax2.scatter([ideal_point[0]], [ideal_point[1]], [ideal_point[2]], s=200, c='yellow',
                    edgecolor='black', marker='*', label='Punkt idealny')
    ax2.set_title("Sfera przekształcona przez G1, G2, G3")
    ax2.set_xlabel("G1")
    ax2.set_ylabel("G2")
    ax2.set_zlabel("G3")
    ax2.legend()

    plt.suptitle("Skalaryzacja liniowa — wersja 3D", fontsize=16)
    plt.tight_layout()
    return fig