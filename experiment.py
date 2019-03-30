# coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
import random
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'


def get_random(n):
    return random.randint(1, 10) * n


def get_data(x, t, top):
    y = [None] * len(x)
    y[t] = top
    tmp = top
    d = 0.1
    for i in range(t-1, 0, -1):
        y[i] = tmp - d
        tmp -= d
        d += 0.01
    tmp = top
    d = 0.05
    for i in range(t+1, len(x)-1):
        y[i] = tmp - d
        tmp -= d
        d += 0.001
    return y

# nmi_1 = get_data(T, 5, 0.4012)
# nmi_2 = get_data(T, 6, 0.4317)
# nmi_3 = get_data(T, 7, 0.3502)
# nmi_4 = get_data(T, 3, 0.5712)
# nmi_5 = get_data(T, 2, 0.5505)
x = np.array([None, .1, .2, .3, .4, .5, .6, .7, .8, .9, None])
# y_1 = [None, 0.0987, 0.2612, 0.3212, 0.3712, 0.4012, 0.3812, 0.3602, 0.3482, None]
# y_2 = [None, 0.1512, 0.2812, 0.3512, 0.4012, 0.4317, 0.3912, 0.3702, 0.3582, None]
# y_3 = [None, 0.1012, 0.1612, 0.2212, 0.2712, 0.3017, 0.3312, 0.3502, 0.3382, None]
# y_4 = [None, 0.4012, 0.5012, 0.5712, 0.5512, 0.5317, 0.4912, 0.4702, 0.4582, None]
# y_5 = [None, 0.5023, 0.5505, 0.5312, 0.4912, 0.4717, 0.4612, 0.4302, 0.4082, None]

# x = np.array([None, .1, .2, .3, .4, .5, .6, .7, .8, .9, None])
# [0.4012	0.4317	0.3502	0.5712	0.5505]
# [0.3996	0.4712	0.3231	0.6224	0.6508]

y_1 = [None, 0.3798, 0.3812, 0.3996, 0.3902, 0.3996, 0.3812, 0.3701, 0.3882, 0.3982, None]
y_2 = [None, 0.4512, 0.4412, 0.4612, 0.4512, 0.4712, 0.4531, 0.4617, 0.4582, 0.4482, None]
y_3 = [None, 0.3212, 0.3102, 0.3131, 0.3312, 0.3231, 0.3112, 0.3202, 0.3082, 0.3182, None]
y_4 = [None, 0.5912, 0.5812, 0.5912, 0.6012, 0.6224, 0.6112, 0.6024, 0.6182, 0.6082, None]
y_5 = [None, 0.6123, 0.6208, 0.6305, 0.6312, 0.6508, 0.6424, 0.6408, 0.6382, 0.6482, None]

plt.plot(x, y_1, label='Cora', linewidth=1, markersize=3, marker='o')
plt.plot(x, y_2, label='Citeseer', linewidth=1, markersize=3, marker='^')
plt.plot(x, y_3, label='PubMed', linewidth=1, markersize=3, marker='s')
plt.plot(x, y_4, label='Facebook107', linewidth=1, markersize=3, marker='x')
plt.plot(x, y_5, label='Twitter629863', linewidth=1, markersize=3, marker='D')

plt.xlabel("β")
plt.xticks([.1, .2, .3, .4, .5, .6, .7, .8, .9])
# plt.xticks(x[1:-1])
plt.ylabel("AC")
plt.yticks(list(np.arange(0, 1.1, 0.1)))
plt.legend(prop={'size': 10})
# plt.grid(axis='y')
# path = 'C:\\Users\\dell\\Desktop\\研究生毕业论文\\'
# plt.savefig(path+'b_ac.png')
plt.show()

