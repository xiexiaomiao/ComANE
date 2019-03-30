import numpy as np


class Node:
    def __init__(self, attribute, structure, node_id=-1):
        self.attribute = attribute
        self.structure = structure
        self.node_id = node_id


def distance(u, v, alpha=0.5):
    return _euclidean_(u.attribute, v.attribute) + alpha * _euclidean_(u.structure, v.structure)


def _euclidean_(v1, v2):
    euclidean = np.sqrt(np.sum(np.square(v1 - v2)))
    return euclidean


def _k_init_(k, nodes):
    n_sample = len(nodes)
    centers = set()

    while len(centers) < k:
        random_id = np.random.randint(0, n_sample)
        centers.add(nodes[random_id])

    return centers


def _assign_to_closest_center_(nodes, centers, alpha):
    center2nodes = {}
    for center in centers:
        center2nodes[center] = []
    for node in nodes:
        distances = [distance(node, center, alpha) for center in centers]
        closest_center = centers[distances.index(min(distances))]
        center2nodes[closest_center].append(node)

    return center2nodes


def _update_centers_(center2nodes):
    new_centers = set()
    for center, nodes in center2nodes.items():
        attribute = None
        structure = None

        for node in nodes:
            if not attribute:
                attribute = node.attribute
            else:
                attribute = np.vstack((attribute, node.attribute))
            if not structure:
                structure = node.structure
            else:
                structure = np.vstack((structure, node.structure))

        attribute = np.mean(attribute, axis=0)
        structure = np.mean(structure, axis=0)
        u = Node(attribute, structure)
        new_centers.add(u)

    return new_centers


def cluster(n_cluster, nodes, max_iter=300):
    seeds = _k_init_(n_cluster)
    centers = seeds

    for i in range(max_iter):
        center2nodes = _assign_to_closest_center_(nodes, centers)
        centers = _update_centers_(center2nodes)
    center_id = {}
    id_ = 1
    for center in center2nodes.keys():
        center_id[center] = id_
        id_ += 1
    partirion = {}
    for center, nodes in center2nodes:
        for node in nodes:
            partirion[node.node_id] = center_id[center]

    return partirion

if __name__ == '__main__':
    k = 6
    nodes = None
    cluster(k, nodes)