import torch
import numpy as np
import matplotlib.pyplot as plt

# import os, sys
# print(os.getcwd())    
# print(os.path.dirname(os.path.abspath(__file__)))     

# Prepare dataset
datasets = np.loadtxt('datasets\diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(datasets[:, :-1])
y_data = torch.from_numpy(datasets[:, [-1]])


# Design model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 24)
        self.linear2 = torch.nn.Linear(24, 12)
        self.linear3 = torch.nn.Linear(12, 1)
        self.activate = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
    
model = Model()


# Construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Training cycle
epoch_list = []
loss_list = []
for epoch in range(2000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)  
    print('Epoch:', epoch, 'loss=', loss.item())

    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 1. 获取预测的概率值
y_pred_probs = model(x_data)

# 2. 设定阈值0.5，把概率变成 0 或 1
# ge 是 greater than or equal (>=) 的缩写
prediction = y_pred_probs.ge(0.5).float() 

# 3. 和真实标签 y_data 对比，看看有多少个一样
# (prediction == y_data) 会生成一堆 True/False
# .sum() 把 True 当作 1 加起来，得到猜对的数量
correct = (prediction == y_data).sum().item()

# 4. 计算比例
total = len(y_data)
accuracy = correct / total

print(f'准确率: {accuracy * 100:.2f}%')

# PLT
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()