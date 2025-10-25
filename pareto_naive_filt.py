def get_P_front(X, directions=None):
    """
    Algorytm z filtracją punktów zdominowanych (front Pareto).
    :param X: lista punktów (tupli lub list)
    :param directions: lista kierunków np. ["min", "min", "max"]
    :return: (lista punktów niezdominowanych, liczba porównań punktów, liczba porównań współrzędnych)
    """
    X = X.copy()
    P = []
    compare_points = 0
    compare_coords = 0

    if not X:
        return P, compare_points, compare_coords

    dim = len(X[0])
    if directions is None:
        directions = ["min"] * dim

    def dominates(a, b):
        """Sprawdza, czy punkt a dominuje punkt b."""
        nonlocal compare_coords
        better = False
        for ai, bi, d in zip(a, b, directions):
            compare_coords += 1
            if d == "min":
                if ai > bi:
                    return False
                elif ai < bi:
                    better = True
            else:
                if ai < bi:
                    return False
                elif ai > bi:
                    better = True
        return better

    while X:
        Y = X[0]
        j = 1
        while j < len(X):
            compare_points += 1
            if dominates(Y, X[j]):
                X.pop(j)
            elif dominates(X[j], Y):
                X.pop(0)
                Y = X[j - 1]
                j = 1
            else:
                j += 1

        P.append(Y)

        X = [pt for pt in X if not dominates(Y, pt)]
        X = [pt for pt in X if pt != Y]

        if len(X) == 1:
            P.append(X[0])
            break

    return P, compare_points, compare_coords



# Test data
data = [
    (5, 5), (3, 6), (4, 4), (5, 3), (3, 3),
    (1, 8), (3, 4), (4, 5), (3, 10), (6, 6), (4, 1), (3, 5)
]

front, compare_points, compare_coords = get_P_front(data)
print(front, compare_points, compare_coords)
