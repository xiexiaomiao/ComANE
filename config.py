class Config:
    def __init__(self):
        # load data
        self.path = 'graph/cora'
        self.edge_file = self.path + 'cora.edgelist'
        self.feature_file = self.path + 'cora.feature'
        self.label_file = self.path + 'cora.label'
        self.direct = False
        # parameters for similarity
        self.num_hop = 7
        self.alpha = 0.5
        # parameters for clustering
        self.k = 7

        # hyperparameter
        self.struct = [2708, 1000, 500, 50]
        self.beta = 0.1

        # parameters for training
        self.random_seed = 1
        self.batch_size = 128
        self.max_iters = 100
        self.lr = 1e-4
        self.epoch = 50