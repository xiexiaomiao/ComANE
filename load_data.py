import networkx as nx
import numpy as np
import scipy.sparse as sp
import math


class LoadData:
    def __init__(self, config):
        self.edge_file = config.edge_file
        self.feature_file = config.feature_file
        self.label_file = config.label_file
        if config.direct:
            self.G = nx.DiGraph()
        else:
            self.G = nx.Graph()
        self.hop = config.num_hop
        self.alpha = config.alpha
        self.read_feature()
        self.read_edge()
        self.A = nx.to_numpy_matrix(self.G)
        if self.label_file:
            self.read_label()
        self.get_similarity()
        # self.construct_X_neighbor()
        self.node2id = {}

    def read_feature(self):
        with open(self.feature_file) as f:
            lines = f.readlines()
            features = []
            for i, line in enumerate(lines[0:]):
                line = line.strip().split('\t')
                features.append(line)
            # for i, line in enumerate(lines[1:]):
            #     line = line.strip().split()
            #     self.node2id[line[0]] = i
            #     features.append(line[1:])
            features = np.array(features, dtype=np.float32)

            self.X = features

    def read_edge(self):
        with open(self.edge_file) as f:
            lines = f.readlines()
            edges = []
            for line in lines:
                line = line.strip().split()
                # edge = (self.node2id[line[0]], self.node2id[line[1]])
                edge = (int(line[0]), int(line[1]))
                edges.append(edge)
            self.G.add_edges_from(edges)

    def read_label(self):
        # self.labels = []
        with open(self.label_file) as f:
            lines = f.readlines()
            self.labels = [int(line.strip().split()[1]) for line in lines]
            # for line in lines:
            #     line = line.strip().split()
            #     self.labels.append(int(line[1]))

    def construct_X_neighbor(self):
        self.X_neighbor = np.zeros(self.X.shape)
        for node in self.G.nodes():
            temp = self.X[node]
            for n in self.G.neighbors(node):
                temp = np.vstack((temp, self.X[n]))
        temp = np.mean(temp, axis=0)
        self.X_neighbor[node] = temp

    def get_similarity(self):
        print('building similarity matrix S, T=%s, alpha=%s' % (self.hop, self.alpha))
        S = np.eye(self.X.shape[0])
        # row, col, data = [], [], []
        node_path = dict(nx.all_pairs_shortest_path_length(self.G))

        for source, path in node_path.items():
            for target, hop in path.items():
                if 0 < hop <= self.hop:
                    sim = math.exp(self.alpha * (1 - hop))
                    S[source][target] = sim
        self.S = S
                    # row.append(source)
                    # col.append(target)
                    # data.append(sim)
        # row = np.array(row)
        # col = np.array(col)
        # self.S = sp.coo_matrix((data, (row, col)), shape=(self.N, self.N)).toarray()

    #
    # def construct_target_neighbors(self, mode):
    #     # construct target neighbor feature matrix
    #     print('construct target neighbor feature matrix...')
    #     if mode == 'self':
    #         return self.X
    #
    #     if mode == 'neighbor':
    #         X_target = np.zeros(self.X.shape)
    #         for node in self.G.nodes():
    #             temp = self.X[node]
    #             for n in self.G.neighbors(node):
    #                 temp = np.vstack((temp, self.X[n]))
    #         temp = np.mean(temp, axis=0)
    #         X_target[node] = temp
    #         return X_target
    #
    #     if mode == 'k-hop-neighbor':
    #         X_target = np.zeros(self.X.shape)
    #         node_path = nx.all_pairs_shortest_path_length(self.G)
    #         for source, path in node_path.items():
    #             temp = self.X[source]
    #             for target, hop in path.items():
    #                 if 0 < hop <= self.hop:
    #                     sim = math.exp(self.alpha * (1 - hop))
    #                     temp = np.vstack((temp, self.X[target] * sim))
    #             temp = np.mean(temp, axis=0)
    #             X_target[source] = temp
    #
    #         return X_target