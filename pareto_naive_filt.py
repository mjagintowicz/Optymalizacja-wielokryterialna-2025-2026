def get_P_front(X, directions=None):
    """
    Znajdowanie punktów niezdominowanych (front Pareto) z uwzględnieniem kierunków min/max.
    :param X: lista punktów (tupli)
    :param directions: lista kierunków, np. ["min", "max", "min"]
    :return: (lista punktów niezdominowanych, liczba porównań)
    """
    P = []
    compare_count = 0
    dim = len(X[0])
    X = X.copy()
    total = len(X)
    dims = len(X[0]) if total else 0

    if directions is None:
        directions = ["min"] * dims

    while X:
        Y = X[0]
        idx_curr = 0
        k = 1

        while k < len(X):
            better = worse = 0

            for a, b, d in zip(X[k], Y, directions):
                compare_count += 1

                if d == "min":
                    if a <= b:
                        better += 1
                    if a >= b:
                        worse += 1
                else:
                    if a >= b:
                        better += 1
                    if a <= b:
                        worse += 1

            if better == dim:
                X.pop(k)
            elif worse == dim:
                Y = X[k]
                X.pop(idx_curr)
                idx_curr = k - 1
            else:
                k += 1

        if Y not in P:
            P.append(Y)

        X = [
            pt for pt in X
            if not all(
                (y <= p if d == "min" else y >= p)
                for y, p, d in zip(Y, pt, directions)
            )
        ]

        if len(X) == 1:
            P.append(X[0])
            break

    return P, compare_count



# Test data
data = [
    (5, 5), (3, 6), (4, 4), (5, 3), (3, 3),
    (1, 8), (3, 4), (4, 5), (3, 10), (6, 6), (4, 1), (3, 5)
]

front, comp = get_P_front(data)
print(front, comp)
