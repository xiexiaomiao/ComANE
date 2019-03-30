from loadData import LoadData
from config import Config
from KNNGraph import KNNGraph
from minibatch_kmeans import Minibatch_kmeans
from evaluation import NMI, get_modularity
import keras as K
import networkx as nx
from keras.models import Model
from keras.layers import Input, Dense, Add, Lambda
from sklearn.metrics.pairwise import cosine_similarity
from networkx.linalg.modularitymatrix import modularity_matrix


if __name__ == '__main__':
    def show(embeddings):
        mk = Minibatch_kmeans(config.k, 256, 100)
        M, labels_pred = mk.fit_predict(embeddings)

        # Q
        partition = {}
        for i, label in enumerate(labels_pred):
            partition[i] = label
        print(get_modularity(data.G, partition))
        print('nmi:', NMI(data.labels, labels_pred))

    config = Config()
    data = LoadData(config)
    data_B = modularity_matrix(data.G)
    knn_graph = KNNGraph(10, cosine_similarity)
    knn_graph.build(data.X)
    data_M = nx.to_numpy_matrix(knn_graph.G)
    n, m = data.X.shape
    representation_size = 256

    B = Input(shape=(n,))
    EB = Dense(representation_size, activation='tanh')(B)
    OB = Dense(n, activation='tanh')(EB)
    # pre-train
    B_AE = Model(inputs=B, outputs=OB)
    B_AE.compile(optimizer=K.optimizers.Adam(lr=1e-4), loss='mse')
    B_AE.fit(data.S, data.S, epochs=20, batch_size=256, shuffle=True)

    print('using structure alone')
    generator = Model(inputs=B, outputs=EB)
    embeddings = generator.predict(data.S)
    show(embeddings)

    M = Input(shape=(n,))
    EM = Dense(representation_size, activation='tanh')(M)
    OM = Dense(n, activation='tanh')(EM)
    # pre-train
    M_AE = Model(inputs=M, outputs=OM)
    M_AE.compile(optimizer=K.optimizers.Adam(lr=1e-4), loss='mse')
    M_AE.fit(data_M, data_M, epochs=20, batch_size=256, shuffle=True)

    print('using attribute alone')
    generator = Model(inputs=M, outputs=EM)
    embeddings = generator.predict(data_M)
    show(embeddings)

    H = Add()([EB, EM])
    B_hat = Dense(n, activation='tanh')(H)
    M_hat = Dense(n, activation='tanh')(H)

    AE = Model(inputs=[B, M], outputs=[B_hat, M_hat])
    AE.compile(optimizer=K.optimizers.Adam(lr=1e-4), loss='mse')
    AE.fit([data.S, data_M], [data.S, data_M], epochs=20, batch_size=256, shuffle=True)

    print('initial ')
    generator = Model(inputs=[B, M], outputs=H)
    embeddings = generator.predict([data.S, data_M])
    show(embeddings)

