import numpy as np
import tensorflow as tf
from krank import Krank
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from evaluation import get_modularity, NMI
from sklearn import preprocessing

class Model:
    def __init__(self, config):
        tf.reset_default_graph()
        tf.set_random_seed(config.random_seed)
        self.config = config
        self._init_graph_()

    def _init_graph_(self):
        self.X = tf.placeholder(tf.float32, shape=[None, None])
        self.A = tf.placeholder(tf.float32, shape=[None, None])
        self.M = tf.placeholder(tf.float32, shape=[None, None])

        # Variables
        w_init = tf.contrib.layers.xavier_initializer()
        sizes = self.config.struct
        W = {}
        b = {}
        for i in range(len(sizes) - 1):
            name = '_encoder_' + str(i)
            W[name] = tf.get_variable(name='W' + name, shape=[sizes[i], sizes[i + 1]], dtype=tf.float32,
                                      initializer=w_init)
            b[name] = tf.get_variable(name='b' + name, shape=[sizes[i + 1]], dtype=tf.float32,
                                      initializer=w_init)
        sizes.reverse()

        for i in range(len(sizes) - 1):
            name = '_decoder_' + str(i)
            W[name] = tf.get_variable(name='W' + name, shape=[sizes[i], sizes[i + 1]], dtype=tf.float32,
                                      initializer=w_init)
            b[name] = tf.get_variable(name='b' + name, shape=[sizes[i + 1]], dtype=tf.float32,
                                      initializer=w_init)
        sizes.reverse()

        # encoder
        for i in range(len(sizes) - 1):
            name = '_encoder_' + str(i)
            if i == 0:
                self.H = tf.nn.tanh(tf.matmul(self.X, W[name]) + b[name])
            else:
                self.H = tf.nn.tanh(tf.matmul(self.H, W[name]) + b[name])

        # decoder
        for i in range(len(sizes) - 1):
            name = '_decoder_' + str(i)
            if i == 0:
                # self.newX = tf.nn.sigmoid(tf.matmul(self.H, self.W[name]) + self.b[name])
                self.X_hat = tf.nn.tanh(tf.matmul(self.H, W[name]) + b[name])
            else:
                # self.newX = tf.nn.sigmoid(tf.matmul(self.newX, self.W[name]) + self.b[name])
                self.X_hat = tf.nn.tanh(tf.matmul(self.X_hat, W[name]) + b[name])

        # Loss
        self.loss_ae = tf.reduce_mean(tf.square(self.X_hat - self.X))
        self.loss_cl = tf.reduce_mean(tf.square(self.H - self.M))
        self.loss = self.loss_ae + self.config.beta * self.loss_cl

        # Optimizer.
        self.optimizer_ae = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss_ae)
        self.optimizer_cl = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss_cl)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def pre_train(self, data):
        print('train...')
        total_batch = int(data.X.shape[0] / self.config.batch_size)
        for it in range(self.config.epoch):
            for i in range(total_batch):
                start_index = np.random.randint(0, data.X.shape[0]-self.config.batch_size)
                X_batch = data.X[start_index: start_index+self.config.batch_size]
                A_batch = data.A[start_index: start_index+self.config.batch_size]
                # X_neighbor_batch = data.X_neighbor[start_index: start_index+self.config.batch_size]
                # feed_dict = {self.X: np.concatenate((X_batch, X_neighbor_batch), axis=1)}
                # feed_dict = {self.X: np.concatenate((X_batch, A_batch), axis=1)}
                feed_dict = {self.X: data.A}
                _, cost = self.sess.run((self.optimizer_ae, self.loss_ae), feed_dict)
            # print('epoch %s: cost = %s' % (it+1, cost))

    def train(self, data):
        self.pre_train(data)
        for loop in range(self.config.max_iters):
            embeddings = self.get_embedding(data)
            # print(embeddings.shape)
            # gmm
            gmm = GaussianMixture(self.config.k)
            gmm.fit(embeddings)
            labels_pred = gmm.predict(embeddings)
            print(labels_pred)
            partition = {}
            for v, p in enumerate(labels_pred):
                partition[v] = p
            q = get_modularity(data.G, partition)
            print('loop %s Q: %s' % (loop, q))

            # M =
            #
            # total_batch = data.X.shape[0] // self.config.batch_size
            # for it in range(self.config.epoch):
            #     for i in range(total_batch):
            #         start_index = np.random.randint(0, data.X.shape[0] - self.config.batch_size)
            #         X_batch = data.X[start_index: start_index + self.config.batch_size]
            #         X_neighbor_batch = data.X_neighbor[start_index: start_index + self.config.batch_size]
            #         M_batch = M[start_index: start_index + self.config.batch_size]
            #         feed_dict = {self.X: np.concatenate((X_batch, X_neighbor_batch), axis=1), self.M: M_batch}
            #         _, cost = self.sess.run((self.optimizer, self.loss), feed_dict)
            #     # print('epoch %s: cost = %s' % (it + 1, cost))

    def get_embedding(self, data):
        # feed_dict = {self.X: np.concatenate((data.X, data.A), axis=1)}
        feed_dict = {self.X: data.A}
        return self.sess.run(self.H, feed_dict)

    # def evaluate(self, metric, data):
    #     embeddings = self.get_embedding(data)
    #     labels = data.labels
    #     labels_pred = KMeans(self.config.k).fit_predict(embeddings)
    #     if metric == 'NMI':
    #         nmi = NMI(labels, labels_pred)
    #         print('AEKM nmi:', nmi)
    #     elif metric == 'Q':
    #         partition = {}
    #         for v, p in enumerate(labels_pred):
    #             partition[v] = p
    #         q = get_modularity(data.G, partition)
    #         print('AEKM Q:', q)
    #     else:
    #         print('error')