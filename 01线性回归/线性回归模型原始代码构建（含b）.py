import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 生成w和b的范围
w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(0.0, 4.1, 0.1)


# 创建网格
w_list, b_list = np.meshgrid(w_range, b_range)

# 初始化MSE列表为与网格相同形状的数组
mse_list = np.zeros_like(w_list)

# # 遍历所有(w, b)组合计算MSE
for i, w in enumerate(w_range):
    print('w = ', w)
    for j, b in enumerate(b_range):
        print('b = ', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        mse = l_sum / 3
        print('MSE = ', mse)
        mse_list[j, i] = mse

print(mse_list)

ax.plot_surface(w_list, b_list, mse_list, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()