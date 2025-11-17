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