def algorithm_filter_after(X):

    P = []
    i = 0
    while X:
        Y = X[0]
        fl = 0
        j = 1
        while j < len(X):
            if all(y <= x for y in Y for x in X[j]):
                X.remove(X[j])
            elif all(x <= y for y in Y for x in X[j]):
                X.remove(Y)
                Y = X[j - 1]
                fl = 1
            else:
                j += 1
                if len(X) == 2:
                    P.append(X[0])
                    P.append(X[1])

        # sprawdzenie pozostałych elementów w pętli zewnętrznej
        j = 0
        while j < len(X) and len(X) > 2:
            if all(y <= x for y in Y for x in X[j]) and Y != X[j]:
                X.remove(X[j])
            elif all(x <= y for y in Y for x in X[j]) and Y != X[j]:
                X.remove(Y)
            else:
                j += 1

        i += 1  # liczba iteracji
        if Y not in P and fl == 1:
            P.append(Y)
        if fl == 0:
            X.remove(Y)

    return P


X_test = [(5,5), (3,6), (4,4), (5,3), (3,3), (1,8), (3,4), (4,5), (3,10), (6,6), (4,1), (3,5)]

print(algorithm_filter_after(X_test))