from loadData import LoadData
from config import Config
import keras as K
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Multiply


conf = Config()
data = LoadData(conf)

# A，X 的维度
size_a = data.A.shape[1]
size_x = data.X.shape[1]

A_i = Input(shape=(size_a,))
X_i = Input(shape=(size_x,))
A_j = Input(shape=(size_a,))
X_j = Input(shape=(size_x,))
ha1i = Dense(500, activation='tanh')(A_i)
hx1i = Dense(500, activation='tanh')(X_i)

hi = Multiply()([ha1i, hx1i])

ha1j = Dense(500, activation='tanh')(A_j)
hx1j = Dense(500, activation='tanh')(X_j)

hj = Multiply()([ha1i, hx1i])

H = Multiply()([hi, hj])
H = Dense(500, activation='tanh')(H)
H = Dense(500, activation='tanh')(H)
out = Dense(1, activation='sigmoid')(H)

model = Model(inputs=[A_i, X_i, A_j, X_j], outputs=[out])
model.compile(optimizer=K.optimizers.Adam(lr=1e-4), loss='mse', metrics=['acc'])

source = []
target = []
weights = []
with open('test_pairs.txt') as f:
    edges = []
    lines = f.readlines()
    for line in lines:
        s, t, w = map(int, line.strip().split())
        source.append(s)
        target.append(t)
        weights.append(w)
# random.shuffle((source, target, weights))
A, X = data.A, data.X
data_A_i = A[source]
data_A_j = A[target]
data_X_i = X[source]
data_X_j = X[target]

model.fit([data_A_i, data_X_i, data_A_j, data_X_j], [np.array(weights)], epochs=100, batch_size=256, shuffle=True)
# test = [data_A_i, data_X_i, data_A_j, data_X_j]
# label_pred = (print(model.predict(test)) > 0.6)
