import numpy as np 
import matplotlib.pyplot as plt 


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 定义前馈线性模型
def forward(x):
    return x * w


# 定义评估模型——即损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 定义两个列表，分别用来记录权重和它的损失值
w_list = []
mse_list = []

# 取权重
for w in np.arange(0.0, 4.0, 0.1):
    print('w = ', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE = ', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

# 绘图可视化结果
plt.plot(w_list, mse_list)
plt.ylabel('loss')
plt.xlabel('w')
plt.show()