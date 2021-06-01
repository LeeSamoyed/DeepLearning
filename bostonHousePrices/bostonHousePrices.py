#
#### 导入需要用到的package
#
import numpy as np
import json

# 前向传播
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost

# 数据load函数
def load_data():
    # 读入训练数据
    datafile = './housing.csv'
    data = np.fromfile(datafile, sep=' ')
    # print(data)

    # 数据处理
    # 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推....
    # 这里对原始数据做reshape，变成N x 14的形式
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS',
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    # print(data)

    # 查看数据
    # x = data[0]
    # print(x.shape)
    # print(x)

    # 数据集划分
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    # print(training_data.shape)

    # 归一化处理
    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = \
                         training_data.max(axis=0), \
                         training_data.min(axis=0), \
         training_data.sum(axis=0) / training_data.shape[0]
    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

# 获取数据
training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]

# 查看数据
# (x[0])
# print(y[0])

# 模型设计
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
w = np.array(w).reshape([13, 1])
x1=x[0]
t = np.dot(x1, w)
# print(t)
b = -0.2 # bias
z = t + b
# print(z)

# 单个测试Loss
# net = Network(13)
# x1 = x[0]
# y1 = y[0]
# z = net.forward(x1)
# # print(z)
# Loss = (y1 - z)*(y1 - z)
# # print(Loss)

# 损失函数计算过程展示
# net = Network(13)
# # 此处可以一次性计算多个样本的预测值和损失函数
# x1 = x[0:3]
# y1 = y[0:3]
# z = net.forward(x1)
# print('predict: ', z)
# loss = net.loss(z, y1)
# print('loss:', loss)

# 梯度下降
net = Network(13)
losses = []
#只画出参数w5和w9在区间[-160, 160]的曲线部分，以及包含损失函数的极值
w5 = np.arange(-160.0, 160.0, 1.0)
w9 = np.arange(-160.0, 160.0, 1.0)
losses = np.zeros([len(w5), len(w9)])
#计算设定区域内每个参数取值所对应的Loss
for i in range(len(w5)):
    for j in range(len(w9)):
        net.w[5] = w5[i]
        net.w[9] = w9[j]
        z = net.forward(x)
        loss = net.loss(z, y)
        losses[i, j] = loss
#使用matplotlib将两个变量和对应的Loss作3D图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
w5, w9 = np.meshgrid(w5, w9)
ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
plt.show()