import numpy as np


class Minibatch_kmeans:
    def __init__(self, k, b, t):
        self.k = k
        self.b = b
        self.t = t

    def similarity(self, v1, v2):
        return -np.sqrt(np.sum(np.square(v1 - v2)))

    def fit_predict(self, X, C=None, init=True):
        N, M = X.shape
        if init:
            randIndex = np.random.randint(0, N, self.k)
            C = X[randIndex]

        for i in range(self.t):
            start_index = np.random.randint(0, N)
            M = X[start_index: start_index + self.b]
            d = {}
            v = {}
            for x in M:
                s = [self.similarity(x, c) for c in C]
                d[tuple(x)] = s.index(max(s))

            for x in M:
                if d[tuple(x)] not in v:
                    v[d[tuple(x)]] = 1
                else:
                    v[d[tuple(x)]] += 1
                lr = 1 / v[d[tuple(x)]]
                C[d[tuple(x)]] = (1 - lr) * C[d[tuple(x)]] + lr * x

        labels_pred = []
        for x in X:
            s = [self.similarity(x, c) for c in C]
            labels_pred.append(s.index(max(s)))

        return C, labels_pred