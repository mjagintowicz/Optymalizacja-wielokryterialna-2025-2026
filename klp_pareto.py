def dominates_dir(a, b, directions):
    """
    Sprawdza czy punkt a dominuje punkt b w zadanych kierunkach.
    Zwraca True/False oraz liczbę porównań współrzędnych wykonanych w tym sprawdzeniu.
    """
    compare_coords = 0
    better_or_equal = True
    strictly_better = False
    for ai, bi, d in zip(a, b, directions):
        compare_coords += 1
        if d == "min":
            if ai > bi:
                better_or_equal = False
                break
            if ai < bi:
                strictly_better = True
        else:  # "max"
            if ai < bi:
                better_or_equal = False
                break
            if ai > bi:
                strictly_better = True
    return better_or_equal and strictly_better, compare_coords


def klp_pareto(points, directions=("min", "min")):
    """
    KLP z kierunkami, liczeniem porównań punktów i współrzędnych.
    Zwraca: front Pareto, liczba porównań punktów, liczba porównań współrzędnych
    """
    compare_points = 0
    compare_coords = 0

    def recursive_klp(X):
        nonlocal compare_points, compare_coords
        if len(X) <= 1:
            return X

        rev = directions[0] == "max"
        X = sorted(X, key=lambda x: x[0], reverse=rev)
        mid = len(X) // 2
        L, R = X[:mid], X[mid:]

        frontL = recursive_klp(L)
        frontR = recursive_klp(R)

        filteredR = []
        for r in frontR:
            dominated = False
            for l in frontL:
                compare_points += 1
                dom, coords = dominates_dir(l, r, directions)
                compare_coords += coords
                if dom:
                    dominated = True
                    break
            if not dominated:
                filteredR.append(r)

        combined = frontL + filteredR
        final_front = []
        for p in combined:
            dominated = False
            for q in combined:
                if p == q:
                    continue
                compare_points += 1
                dom, coords = dominates_dir(q, p, directions)
                compare_coords += coords
                if dom:
                    dominated = True
                    break
            if not dominated and p not in final_front:
                final_front.append(p)
        return final_front

    front = recursive_klp(points)
    return front, compare_points, compare_coords

data = [
    (5,5), (3,6), (4,4), (5,3), (3,3),
    (1,8), (3,4), (4,5), (3,10), (6,6), (4,1), (3,5)
]

front, compare_points, compare_coords = klp_pareto(data)
print("Front Pareto:", front, compare_points, compare_coords)