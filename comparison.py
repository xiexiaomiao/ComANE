import igraph as ig
import numpy as np
from config import Config
from loadData import LoadData
from evaluation import get_modularity, NMI
from sklearn.cluster import KMeans
from krank import Krank
import matplotlib.pyplot as plt
# from SDAE.sdae import StackedDenoisingAE

conf = Config()
data = LoadData(conf)
nodes = data.G.nodes()
edges = data.G.edges()
labels_true = data.labels

# g = ig.Graph().as_directed()
g = ig.Graph()
g.add_vertices(nodes)
g.add_edges(edges)
print(g.summary())

info = g.community_infomap()
nmi = NMI(labels_true, info.membership)
print('infomap:', nmi)
print('infomap:', info.modularity)

lpa = g.community_label_propagation()
nmi = NMI(labels_true, lpa.membership)
print('LPA:', nmi)
print('LPA:', lpa.modularity)

kmeans = KMeans(conf.k)
labels_pred = kmeans.fit_predict(data.X)
nmi = NMI(labels_true, labels_pred)
print('kmeans:', nmi)
