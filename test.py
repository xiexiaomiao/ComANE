from ASNE.LoadData import LoadData
from ASNE.LoadMyData import LoadMyData
from ASNE.SNE import SNE
from ASNE.MySNE import MySNE
from ASNE.evaluation import get_modularity
from sklearn import mixture
import utils.IO_utils as io_utils
import numpy as np
import networkx as nx
import tensorflow as tf
import pickle

num_iter = 5
# ks = [5, 10, 15, 20]
ks = [5, 10, 15]
betas = [.1, .2, .3, .4, .5]
reg_covar = 0.00001

lr = 0.001
id_embedding_size = 64
attr_embedding_size = 64

path = './data/M10/'
data = LoadMyData(path, random_seed=1)
nodes = data.nodes
links = data.links
nodes_id = data.nodes['node_id']

G = nx.Graph()
G.add_nodes_from(nodes_id)
for i in range(len(links)):
    e = (data.node_map[links[i][0]], data.node_map[links[i][1]])
    G.add_edge(*e)
# G.add_edges_from(links)
# G = nx.read_edgelist('../data/M10/doublelink.edgelist', delimiter=' ')

# come_embeddings = io_utils.load_embedding(
#             file_name='UNC_alpha-0.1_beta-0.1_ws-10_neg-5_lr-0.025_icom-37_ind-37_k-2_ds-0.0', path='./data')
for k in ks:
    for beta in betas:
        with open('./data/m10_experiment.txt', 'a') as f:
            f.writelines('k = '+str(k)+', beta = '+str(beta)+'\n')
            # # come发现社区
            # g_mixture = mixture.GaussianMixture(n_components=k, reg_covar=0, covariance_type='full', n_init=10)
            # g_mixture.fit(come_embeddings)
            # centroid = g_mixture.means_.astype(np.float32)
            # covariance_mat = g_mixture.covariances_.astype(np.float32)
            # inv_covariance_mat = g_mixture.precisions_.astype(np.float32)
            # pi = g_mixture.predict_proba(come_embeddings).astype(np.float32)
            # # print(centroid.shape, covariance_mat.shape, pi.shape)
            #
            # # 计算模块度
            # part = {}
            # for node in nodes_id:
            #     # print(np.where(pi[node] == np.max(pi[node]))[0])
            #     part[str(node+1)] = int(np.where(pi[node] == np.max(pi[node]))[0])
            # # print(part)
            # modularity = get_modularity(G, part)
            # f.writelines('come_modularity = ' + "{:.9f}".format(modularity) + '\n')


            # sne发现社区
            node_learner = SNE(data, id_embedding_size, attr_embedding_size)
            node_learner.train()
            Embeddings_out_sne = node_learner.getEmbedding('out_embedding', nodes)
            Embeddings_in_sne = node_learner.getEmbedding('embed_layer', nodes)
            Embeddings_sne = Embeddings_out_sne + Embeddings_in_sne
            # sne_embeddings = sne_node_learner.getEmbedding('embed_layer', sne_node_learner.nodes)
            # io_utils.save_embedding(sne_embeddings, nodes_id, file_name='sne_embeddings', path=path)

            Embeddings = Embeddings_sne
            # Embeddings = io_utils.load_embedding(file_name='asne_embeddings', path=path)

            for it in range(num_iter):
                g_mixture = mixture.GaussianMixture(n_components=k, reg_covar=reg_covar, n_init=10)
                g_mixture.fit(Embeddings)
                centroid = g_mixture.means_.astype(np.float32)
                covariance_mat = g_mixture.covariances_.astype(np.float32)
                inv_covariance_mat = g_mixture.precisions_.astype(np.float32)
                pi = g_mixture.predict_proba(Embeddings).astype(np.float32)
                # print(centroid.shape, covariance_mat.shape, pi.shape)

                # 计算模块度
                part = {}
                for node in nodes_id:
                    part[str(node+1)] = int(np.where(pi[node] == np.max(pi[node]))[0])
                modularity = get_modularity(G, part)
                if it == 0:
                    f.writelines("sne_modularity = "+"{:.9f}".format(modularity) + '\n')
                else:
                    f.writelines("epoch:"+str(it)+" my_modularity = "+"{:.9f}".format(modularity)+'\n')
                node_learner = MySNE(data, id_embedding_size, attr_embedding_size, centroid=centroid,
                                     covariance_mat=covariance_mat, pi=pi, k=k, beta=beta, lr=lr)
                node_learner.train()
                Embeddings_out = node_learner.getEmbedding('out_embedding', node_learner.nodes)
                Embeddings_in = node_learner.getEmbedding('embed_layer', node_learner.nodes)
                Embeddings = Embeddings_out + Embeddings_in
                # Embeddings = my_node_learner.getEmbedding('embed_layer', my_node_learner.nodes)

            # io_utils.save_embedding(Embeddings, nodes_id, file_name='my_embeddings', path=path)
