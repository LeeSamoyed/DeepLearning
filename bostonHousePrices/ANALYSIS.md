#过程解析

##数据加载
```python
# 读书数据后在load函数中将进行数据处理
# 读书数据 -> 修改维度 -> 数据集划分 -> 归一化处理（归一化的最大值最小值和均值以训练用数据集为准，测试数据也用训练的数据去处理）
train_data, test_data = load_data()
```

## 初始化网络
```python
# 初始化 w,b 两个参数
self.w = np.random.randn(num_of_weights, 1)
self.b = 0.

# 前向传播
# 前向传播就是简单的累加 z = x*w + b （这里的x是一个向量）
def forward(self, x)

# loss计算
# 均方差损失函数 (z-y)^2累加之后求均值
def loss(self, z, y)

# 梯度计算
# 理解公式
def gradient(self, x, y)
    
# 步长的理解
def update(self, gradient_w, gradient_b, eta=0.01)

# 训练函数整合
def train(self, training_data, num_epochs, batch_size=10, eta=0.01)
```

## 测试思路
```python
# 就是直接把gradient_x和gradient_b保存下来，等到测试的时候直接用 x*w+b 就可以返回测试值了
```