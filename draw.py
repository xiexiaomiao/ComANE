import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
from networkx.drawing import nx_pylab

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# my_font = font_manager.FontProperties(fname='C:/Windows/Fonts/times.ttf')

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'


# x1 = [0.05, 0.26, 0.41, 0.49, 0.54, 0.55, 0.63, 0.61]
# x2 = [0.02, 0.25, 0.45, 0.52, 0.56, 0.63, 0.64, 0.61]
# x3 = [0.30, 0.41, 0.42, 0.46, 0.44, 0.48, 0.45, 0.44]
# x4 = [0.11, 0.26, 0.31, 0.36, 0.38, 0.37, 0.36, 0.35]

# y = [1, 2, 3, 4, 5, 6, 7, 8]
# # plt.title('Q value with different T in four dataset')
# plt.plot(y, x1, color='green', label='citeseer')
# plt.plot(y, x2, color='red', label='cora')
# plt.plot(y, x3,  color='skyblue', label='facebook')
# plt.plot(y, x4, color='blue', label='UNC')
# plt.legend() # 显示图例
#
# plt.xlabel('T')
# plt.ylabel('Q')
# plt.show()

# x1 = [0.47, 0.53, 0.58, 0.60, 0.63, 0.63, 0.61, 0.58]
# x2 = [0.29, 0.39, 0.48, 0.58, 0.64, 0.56, 0.53, 0.52]
# x3 = [0.34, 0.39, 0.37, 0.41, 0.45, 0.44, 0.44, 0.43]
# x4 = [0.21, 0.25, 0.29, 0.31, 0.36, 0.35, 0.33, 0.32]
#
# y = [.1, .2, .3, .4, .5, .6, .7, .8]
# # plt.title('Q value with different T in four dataset')
# plt.plot(y, x1, color='green', label='citeseer')
# plt.plot(y, x2, color='red', label='cora')
# plt.plot(y, x3,  color='skyblue', label='facebook')
# plt.plot(y, x4, color='blue', label='UNC')
# plt.legend() # 显示图例
# my_y_ticks = np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)
# plt.xlabel('α')
# plt.ylabel('Q')
# plt.show()

# x1 = [0.63, 0.60, 0.59, 0.57, 0.53, 0.45, 0.41, 0.38]
# x2 = [0.58, 0.60, 0.63, 0.64, 0.61, 0.60, 0.53, 0.51]
# x3 = [0.60, 0.63, 0.66, 0.67, 0.65, 0.61, 0.58, 0.55]
# x4 = [0.39, 0.37, 0.34, 0.31, 0.30, 0.31, 0.29, 0.27]
#
# y = [.1, .2, .3, .4, .5, .6, .7, .8]
# # plt.title('Q value with different T in four dataset')
# plt.plot(y, x1, color='green', label='citeseer')
# plt.plot(y, x2, color='red', label='cora')
# plt.plot(y, x3,  color='skyblue', label='facebook')
# plt.plot(y, x4, color='blue', label='UNC')
# plt.legend() # 显示图例
# my_y_ticks = np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)
# plt.xlabel('β')
# plt.ylabel('Q')
# plt.show()

# x1 = [None, 0.2, 0.45, 0.57, 0.42, 0.38, 0.36, 0.35, 0.34, None]
# x2 = [None, 0.25, 0.51, 0.62, 0.54, 0.47, 0.45, 0.44, 0.42, None]
# y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# plt.plot(y, x1, color='red', label='NMI', marker='*')
# plt.plot(y, x2,  color='blue', label='AC', marker='.')
# plt.legend()
# my_y_ticks = np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)
# plt.xlabel('T')
# plt.show()

# x1 = [None, 0.38, 0.57, 0.53, 0.49, 0.45, 0.41, 0.38, 0.36, None]
# x2 = [None, 0.5, 0.62, 0.60, 0.58, 0.56, 0.54, 0.53, 0.52, None]
# y = [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
# plt.plot(y, x1, color='red', label='NMI', marker='*')
# plt.plot(y, x2,  color='blue', label='AC', marker='.')
# plt.legend()
# # my_x_ticks = np.arange(0, 3.5, 0.5)
# # plt.xticks(my_x_ticks)
# my_y_ticks = np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)
# plt.xlabel('λ')
# plt.show()

# G = nx.karate_club_graph()
# c = list(kernighan_lin_bisection(G))
# node_colors = ["blue" if n in c[0] else "red" for n in G.nodes()]
# pos = nx.spring_layout(G)
# # plt.figure(figsize=(20, 8), dpi=80)
# nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors)
# nx.draw_networkx_edges(G, pos=pos)
#
# # plt.title("图1-1：扎克空手道俱乐部数据集")
# plt.axis('off')
# plt.savefig("C:/Users/dell/Desktop/研究生毕业论文/1.png")
# plt.show()

# MAE = [1.0750, 0.9654, 0.8475, 0.8953, 0.8026, ]
# RMSE = [1.4075, 1.2799, 1.1516, 1.1117, 1.0811]
# label_list = ['Deepwalk', 'HeteMF', 'HeRec', 'NCF', 'HINCF']  # 横坐标刻度显示值
# x = range(len(MAE))
#
# # label_list = ['2014', '2015', '2016', '2017']    # 横坐标刻度显示值
# # num_list1 = [20, 30, 15, 35]      # 纵坐标值1
# # num_list2 = [15, 30, 40, 20]      # 纵坐标值2
# # x = range(len(num_list1))
# """
# 绘制条形图
# left:长条形中点横坐标
# height:长条形高度
# width:长条形宽度，默认值0.8
# label:为后面设置legend准备
# """
# rects1 = plt.bar(left=x, height=MAE, width=0.4, alpha=0.8, color='#7EC0EE', label="MAE")
# # rects2 = plt.bar(left=[i + 0.4 for i in x], height=RMSE, width=0.4, color='green', label="RMSE")
# plt.ylim(0, 1.5)  # y轴取值范围
# plt.ylabel("MAE")
# """
# 设置x轴刻度显示值
# 参数一：中点坐标
# 参数二：显示值
# """
# plt.xticks([index + 0.2 for index in x], label_list)
# # plt.xlabel("MAE")
# # plt.title("某某公司")
# # plt.legend()  # 设置题注
# # 编辑文本
# # for rect in rects1:
# #     height = rect.get_height()
# #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
# # for rect in rects2:
# #     height = rect.get_height()
# #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
# plt.grid(axis="y")
# plt.show()

# 实验结果对比图画柱形图，一行一幅图。类似下图，宽度最好比下图窄，字体全部用Times New Roman.
#
# 实验对比结果
def draw_bar(x_label, height, y_label, filename):
    x = range(len(height))
    plt.bar(left=x, height=height, width=0.4, alpha=0.8, color='#9FB6CD')
    plt.ylim(0, 1)  # y轴取值范围
    plt.ylabel(y_label)
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks(x, x_label)
    # plt.xlabel("年份")
    # plt.title("某某公司")
    # plt.legend()  # 设置题注
    plt.grid(axis="y")
    # plt.show()
    plt.savefig(filename)


def draw_plot(x, y, x_label, y_label, filename):
    plt.plot(x, y, c='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)


labels = ['Deepwalk', 'HeteMF', 'HeRec', 'NCF', 'HINCF']
MAE_yelp = [1.0750, 0.9654, 0.8475, 0.8953, 0.8026]
RMSE_yelp = [1.4075, 1.2799, 1.1516, 1.1117, 1.0811]

MAE_mvl = [0.5703, 0.5750, 0.5519, 0.5515, 0.5489]
RMSE_mvl = [0.7446, 0.7556, 0.7053, 0.7121, 0.6982]

# draw_bar(labels, MAE_yelp, 'MAE', 'C://Users//dell//Desktop//shiyantu//mae_yelp.png')
# draw_bar(labels, RMSE_yelp, 'RMSE', 'C://Users//dell//Desktop//shiyantu//rmse_yelp.png')
# draw_bar(labels, MAE_mvl, 'MAE', 'C://Users//dell//Desktop//shiyantu//mae_mv.png')
# draw_bar(labels, RMSE_mvl, 'RMSE', 'C://Users//dell//Desktop//shiyantu//rmse_mv.png')

# 一、参数alpha
# Yelp
alpha = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
MAE_yelp_alpha = [0.8258, 0.8241, 0.8235, 0.8229, 0.8226, 0.8239]
RMSE_yelp_alpha = [1.1061, 1.1057, 1.1040, 1.1027, 1.1011, 1.1030]

MAE_mvl_alpha = [0.5619, 0.5597, 0.5530, 0.5520, 0.5518, 0.5529]
RMSE_mvl_alpha = [0.7032, 0.7025, 0.7018, 0.7012, 0.7013, 0.7015]
# draw_plot(alpha, MAE_yelp_alpha, 'α', 'MAE', 'C://Users//dell//Desktop//shiyantu//mae_y_a.png')
# draw_plot(alpha, RMSE_yelp_alpha, 'α', 'RMSE', 'C://Users//dell//Desktop//shiyantu//rmse_y_a.png')
# draw_plot(alpha, MAE_mvl_alpha, 'α', 'MAE', 'C://Users//dell//Desktop//shiyantu//mae_mv_a.png')
# draw_plot(alpha, RMSE_mvl_alpha, 'α', 'RMSE', 'C://Users//dell//Desktop//shiyantu//rmse_mv_a.png')

# 二、参数epochs
# Yelp
epochs = [100, 200, 300, 400, 500]
MAE_yelp_epoch = [0.8240, 0.8237, 0.8242, 0.8250, 0.8259]
RMSE_yelp_epoch = [1.1020, 1.1014, 1.1029, 1.1035, 1.1040]

MAE_mvl_epoch = [0.5495, 0.5490, 0.5491, 0.5509, 0.5517]
RMSE_mvl_epoch = [0.6990, 0.6985, 0.6984, 0.6991, 0.6999]

# draw_plot(epochs, MAE_yelp_epoch, 'epoch', 'MAE', 'C://Users//dell//Desktop//shiyantu//mae_y_e.png')
# draw_plot(epochs, RMSE_yelp_epoch, 'epoch', 'RMSE', 'C://Users//dell//Desktop//shiyantu//rmse_y_e.png')
# draw_plot(epochs, MAE_mvl_epoch, 'epoch', 'MAE', 'C://Users//dell//Desktop//shiyantu//mae_mv_e.png')
# draw_plot(epochs, RMSE_mvl_epoch, 'epoch', 'RMSE', 'C://Users//dell//Desktop//shiyantu//rmse_mv_e.png')

# 三、参数batch_size
batch_size = [64, 128, 256, 512, 1024]
MAE_yelp_b = [0.8340, 0.8335, 0.8328, 0.8332, 0.8337]
RMSE_yelp_b = [1.1120, 1.1117, 1.1112, 1.1119, 1.1121]

MAE_mvl_b = [0.5599, 0.5590, 0.5588, 0.5591, 0.5600]
RMSE_mvl_b = [0.7092, 0.7088, 0.7081, 0.7085, 0.7090]

# draw_plot(batch_size, MAE_yelp_b, 'batch_size', 'MAE', 'C://Users//dell//Desktop//shiyantu//mae_y_b.png')
# draw_plot(batch_size, RMSE_yelp_b, 'batch_size', 'RMSE', 'C://Users//dell//Desktop//shiyantu//rmse_y_b.png')
# draw_plot(batch_size, MAE_mvl_b, 'batch_size', 'MAE', 'C://Users//dell//Desktop//shiyantu//mae_mv_b.png')
# draw_plot(batch_size, RMSE_mvl_b, 'batch_size', 'RMSE', 'C://Users//dell//Desktop//shiyantu//rmse_mv_b.png')