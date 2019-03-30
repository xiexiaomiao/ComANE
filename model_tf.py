import tensorflow as tf
import numpy as np
from config import Config
from loadData import LoadData


class Model():
    def __init__(self, conf):
        self.struct = conf.struct
        self._init_graph_()

    def _init_graph_(self, n_unit, input):
        W = tf.get_variable("W", shape=[784, 256],
                            initializer=tf.contrib.layers.xavier_initializer())
        return