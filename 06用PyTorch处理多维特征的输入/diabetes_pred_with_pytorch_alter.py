import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # sklearn库中专用于数据集划分的包


# Prepare dataset
dataset = np.loadtxt('datasets\diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = dataset[:, :-1]
y_data = dataset[:, [-1]]

# Train set : Test set = 8 : 2
x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train = torch.from_numpy(x_train_np)
x_test = torch.from_numpy(x_test_np)
y_train = torch.from_numpy(y_train_np)
y_test = torch.from_numpy(y_test_np)
print(f"训练集大小：{len(x_train)}, 测试集大小：{len(x_test)}")


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
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)  
    # print('Epoch:', epoch, 'loss=', loss.item())

    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


y_train_probs = model(x_train)
prediction_train = y_train_probs.ge(0.5).float()
correct_train = (prediction_train == y_train).sum().item()
total_train = len(y_train)
accuracy_train = correct_train / total_train
print(f'训练集准确率: {accuracy_train * 100:.2f}%')

y_test_probs = model(x_test)
prediction = y_test_probs.ge(0.5).float()
correct = (prediction == y_test).sum().item()
total = len(y_test)
accuracy = correct / total
print(f'测试集准确率: {accuracy * 100:.2f}%')


# PLT
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()