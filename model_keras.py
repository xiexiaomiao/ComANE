from loadData import LoadData
from config import Config
from evaluation import get_modularity, NMI
import keras as K
from keras.models import Model
from keras.layers import Input, Dense, Add, Lambda
from minibatch_kmeans import Minibatch_kmeans

conf = Config()
data = LoadData(conf)

# citeseer: 3312 3703, cora: 2708 1432, facebook: 4039 1406, unc: 18163 2788
r_size = 200
beta = 0.1
lam = 0.1
k = 6
s_size, x_size = data.X.shape

S = Input(shape=(s_size,))
X = Input(shape=(x_size,))

# encoder
encoded_S = Dense(r_size, activation='tanh')(S)
# encoded_S = Dense(100, activation='tanh')(encoded_S)
encoded_X = Dense(r_size, activation='tanh')(X)
# encoded_X = Dense(100, activation='tanh')(encoded_X)

H = Add()([Lambda(lambda x: x * beta)(encoded_X), Lambda(lambda x: x * (1 - beta))(encoded_S)])
# decoder
# decoded_S = Dense(200, activation='tanh')(H)
S_hat = Dense(s_size, activation='tanh')(H)
# decoded_X = Dense(200, activation='tanh')(H)
X_hat = Dense(x_size, activation='tanh')(H)

ae = Model(inputs=[S, X], outputs=[S_hat, X_hat])
ae.compile(optimizer=K.optimizers.Adam(lr=1e-4), loss='mse')
ae.fit([data.S, data.X], [data.S, data.X], epochs=100, batch_size=256, shuffle=True)
encoder = Model(inputs=[S, X], outputs=H)
R = encoder.predict([data.S, data.X])
mk = Minibatch_kmeans(k, 256, 100)
M, labels_pred = mk.fit_predict(R)
# res = []
# partition = {}
# for i, label in enumerate(labels_pred):
#     partition[i] = label
# print('initial: T=%s %s\n' % (conf.num_hop, get_modularity(data.G, partition)))
# res.append(get_modularity(data.G, partition))
print('nmi:', NMI(data.labels, labels_pred))
# for i in range(10):
#     ae = Model(inputs=[S, X], outputs=[S_hat, X_hat, H])
#     ae.compile(optimizer=K.optimizers.Adam(lr=1e-4), loss='mse', loss_weights=[1, 1, 0.1])
#     ae.fit([data.S, data.X], [data.S, data.X, M[labels_pred]], epochs=100, batch_size=256, shuffle=True)
#     encoder = Model(inputs=[S, X], outputs=H)
#     R = encoder.predict([data.S, data.X])
#     M, labels_pred = mk.fit_predict(R, M, init=False)
#     partition = {}
#     for i, label in enumerate(labels_pred):
#         partition[i] = label
#     res.append(get_modularity(data.G, partition))
#     print('loop %s: %s' % (i+1, get_modularity(data.G, partition)))
#
# print(res)
# f.close()

