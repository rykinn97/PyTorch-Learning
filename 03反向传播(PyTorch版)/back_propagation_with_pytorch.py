import torch


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('Predict (before training):', 4, forward(4.0).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        loss_val.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    print('Epoch:', epoch, 'loss = ', loss_val.item())

print('Predict (after training):', 4, forward(4.0).item())

'''
    用PyTorch实现反向传播过程:
    tensor类型的数据中,.data保存权重,.grad保存梯度
    先将权重创建为张量,后续所有计算只要有张量的参与,其余数据和结果自动转化为张量类型
    求出模型损失值后要紧接着开启loss.backward(),该函数的作用是求出反向传播所有的权重梯度
    对于一个权重w来说,w本身是张量无法直接计算,必须转为w.data才能计算,同时对于梯度更新的梯度,w.grad也无法直接计算,要转为w.grad.data
    如果要单独取值,要使用.item()标量函数
    切记:在该模型构建中,更新完梯度要紧跟着将梯度信息清零:w.grad.data.zero_()
'''
