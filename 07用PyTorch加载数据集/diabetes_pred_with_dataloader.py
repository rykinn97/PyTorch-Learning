import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Prepare dataset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
dataset = DiabetesDataset(r'datasets\diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)
print(len(train_loader))
print(train_loader.batch_size)


# Design Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 24)
        self.linear2 = torch.nn.Linear(24, 12)
        self.linear3 = torch.nn.Linear(12, 1)
        self.activate = torch.nn.ReLU()
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


epoch_list = []
loss_list = []

# Training cycle
if __name__ == '__main__':
    for epoch in range(100):
        loss_total = 0
        loss_avg = 0.0
        correct_total = 0
        total_sample = 0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss_total += loss.item()

            # 准确率
            predicted = y_pred.ge(0.5).float()
            correct_total += (predicted == labels).sum().item()
            total_sample += labels.size(0)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_avg = loss_total / len(train_loader)
        acc_avg = correct_total / total_sample
        epoch_list.append(epoch)
        loss_list.append(loss_avg)
        print('Epoch:', epoch, 'loss = ', loss_avg, 'acc = ', acc_avg)


plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(visible=True)
plt.show()
