def algorithm_no_filter(X, directions=None):
    """
    Algorytm wyszukiwania punktów niezdominowanych (bez filtrowania),
    minimalna modyfikacja oryginalnego kodu.
    :param X: lista punktów (tupli)
    :param directions: lista kierunków ["min"/"max"] dla każdego wymiaru
    :return: (P, compare_count)
    """
    P = []
    X = X.copy()
    i = 0
    compare_count = 0
    n_dims = len(X[0]) if X else 0
    if directions is None:
        directions = ["min"] * n_dims

    while X:
        Y = X[0]
        fl = 0
        j = 1
        while j < len(X):
            better = worse = 0
            for y_val, x_val, d in zip(Y, X[j], directions):
                compare_count += 1
                if d == "min":
                    if y_val <= x_val:
                        better += 1
                    if y_val >= x_val:
                        worse += 1
                else:  # "max"
                    if y_val >= x_val:
                        better += 1
                    if y_val <= x_val:
                        worse += 1

            if better == n_dims:
                X.pop(j)
            elif worse == n_dims:
                X.pop(0)
                Y = X[j-1]
                fl = 1
            else:
                j += 1
                if len(X) == 2:
                    for pt in X:
                        if pt not in P:
                            P.append(pt)
        i += 1
        if Y not in P and fl == 1:
            P.append(Y)
        if fl == 0 and Y in X:
            X.remove(Y)

    return P, compare_count