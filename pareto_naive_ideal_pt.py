def find_non_dominated_points(X, directions=None):
    'Ideal point distance algorithm with min/max selection per dimension'
    P = []
    compare_points = 0
    compare_coords = 0
    total = len(X)
    dims = len(X[0]) if total else 0

    if directions is None:
        directions = ["min"] * dims

    X_transformed = [
        tuple(-x[i] if directions[i] == "max" else x[i] for i in range(dims))
        for x in X
    ]

    min_vals = [min(pt[i] for pt in X_transformed) for i in range(dims)]

    dist_list = [
        (sum((pt[i] - min_vals[i]) ** 2 for i in range(dims)), idx)
        for idx, pt in enumerate(X_transformed)
    ]
    dist_list.sort()

    checked = set()

    for _, idx in dist_list:
        if idx in checked:
            continue

        base_point = X_transformed[idx]
        P.append(X[idx])

        for j, other in enumerate(X_transformed):
            if j in checked:
                continue
            compare_points += 1
            dominated = True
            for b, o in zip(base_point, other):
                compare_coords += 1
                if b > o:
                    dominated = False
                    break
            if dominated:
                checked.add(j)

    return P, compare_points, compare_coords


# Test data
X = [
    (5, 5), (3, 6), (4, 4), (5, 3), (3, 3),
    (1, 8), (3, 4), (4, 5), (3, 10), (6, 6),
    (4, 1), (3, 5)
]

P, compare_count, compare_coords = find_non_dominated_points(X)
print(P, compare_count, compare_coords)
