def algorithm_no_filter(X):

    """
    :param X: zbiór punktów dwuelementowych
    :return: zbiór punktów niezdominowanych
    """

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
                Y = X[j-1]
                fl = 1
            else:
                j += 1
                if len(X) == 2:
                    P.append(X[0])
                    P.append(X[1])
        i += 1      # liczba iteracji
        if Y not in P and fl == 1:
            P.append(Y)
        if fl == 0:
            X.remove(Y)

    return P
