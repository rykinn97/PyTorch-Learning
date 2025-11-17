import numpy as np 
import torch


# Tensor(张量)的创建方式
# 1. 直接从数据创建,数据类型自动推断
# data1 = [[1, 2], [3, 4], [5, 6]]
# data2 = [1, 2, 3]
# print(type(data1))
# print(data1)
# print(type(data2))
# print(data2)
# x_data1 = torch.tensor(data1)
# x_data2 = torch.tensor(data2)
# print(type(x_data1))
# print(x_data1)
# print(type(x_data2))
# print(x_data2)

# 2. 从NumPy数组创建
# np_array = np.array([[1, 2], [3, 4], [5, 6]])
# print(type(np_array))
# print(np_array)
# x_np = torch.from_numpy(np_array)
# print(type(x_np))
# print(x_np)

# 3. 从另一个张量创建
'''
    介绍了两种从张量创建张量的方式,两种方式都不会直接复制,而是只继承旧张量的属性(形状和数据类型)
    torch.ones_like:继承旧张量的属性,内容用标量1来填充;
    torch.rand_like:继承旧张量的属性,内容用区间[0,1)上的均匀分布随机数来填充
'''
# data1 = [[1, 2], [3, 4], [5, 6]]
# x_data1 = torch.tensor(data1)
# x_ones = torch.ones_like(x_data1)
# x_rand = torch.rand_like(x_data1,  dtype=torch.float)
# print(type(x_data1))
# print(type(x_ones))
# print(type(x_rand))
# print(x_data1)
# print(x_ones)
# print(x_rand)


