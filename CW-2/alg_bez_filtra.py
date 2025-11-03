def algorithm_no_filter(X, directions=None):
    """
    Algorytm wyszukiwania punktów niezdominowanych (bez filtrowania),
    zlicza porównania punktów i współrzędnych oraz obsługuje kierunki.
    :param X: lista punktów (tupli)
    :param directions: lista kierunków ["min"/"max"] dla każdego wymiaru
    :return: (P, compare_points, compare_coords)
    """
    X = X.copy()
    P = []
    compare_points = 0
    compare_coords = 0
    n_o_cord = len(X[0]) if X else 0

    if directions is None:
        directions = ["min"] * n_o_cord

    while len(X) > 0:
        Y = X[0]
        fl = 0
        j = 1
        id_y = 0

        while j < len(X):
            compare_points += 1
            greater = 0
            lower = 0

            for x_val, y_val, d in zip(X[j], Y, directions):
                compare_coords += 1
                if d == "min":
                    if x_val >= y_val:
                        greater += 1
                    if x_val <= y_val:
                        lower += 1
                else:  # "max"
                    if x_val <= y_val:
                        greater += 1
                    if x_val >= y_val:
                        lower += 1

            if greater == n_o_cord:
                X.pop(j)
            elif lower == n_o_cord:
                Y = X[j]
                X.pop(id_y)
                id_y = j - 1
                fl = 1
            else:
                j += 1

        if Y not in P:
            P.append(Y)

        if fl == 0 and Y in X:
            X.remove(Y)

    return P, compare_points, compare_coords


X = [(5,5), (3,6), (4,4), (5,3), (3,3),
     (1,8), (3,4), (4,5), (3,10), (6,6),
     (4,1), (3,5)]

P, pts_cmp, coords_cmp = algorithm_no_filter(X, directions=["min","min"])
print("Front Pareto:", P)
print("Porównania punktów:", pts_cmp)
print("Porównania współrzędnych:", coords_cmp)