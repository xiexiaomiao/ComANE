import igraph as ig
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

from evaluation import *


class Krank:
    def __init__(self, n_cluster, mu):
        self.k = n_cluster
        self.mu = mu

    def similarity(self, vi, vj):
        euclidean = np.sqrt(np.sum(np.square(self.embeddings[vi] - self.embeddings[vj])))
        return 1 / (1 + euclidean)

    def _initial_centroids(self):
        pagerank = nx.pagerank(self.G)
        pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        V = [e[0] for e in pagerank]
        # P = [e[1] for e in pagerank]

        seeds = [V[0]]
        i = 1
        while len(seeds) < self.k:
            flag = True
            for seed in seeds:
                # print(self.similarity(seed, V[i]))
                if self.similarity(seed, V[i]) > self.mu:
                    flag = False
                    break
            if flag:
                seeds.append(V[i])
            i += 1
        return seeds

    def closest_centroid(self, v, centroids):
        S = [self.similarity(v, c) for c in centroids]
        return centroids[S.index(max(S))]

    def updated_centroids(self, clusters):
        centroids = []
        for nodes in clusters.values():
            G_sub = self.G.subgraph(nodes)
            pagerank = nx.pagerank(G_sub)
            pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            centroid = pagerank[0][0]
            centroids.append(centroid)
        return centroids

    def fit_predict(self, G, embeddings):
        self.G = G
        self.embeddings = embeddings
        centroids = self._initial_centroids()
        changed = True

        while changed:
            clusters = {}
            for c in centroids:
                clusters[c] = []

            for v in self.G.nodes():
                c = self.closest_centroid(v, centroids)
                clusters[c].append(v)

            new_centroids = self.updated_centroids(clusters)
            changed = (centroids == new_centroids)
            centroids = new_centroids

        partition = dict()
        for k, v in clusters.items():
            for node in v:
                partition[node] = k

        return centroids, partition