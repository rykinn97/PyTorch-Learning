import torch
import numpy as np
import matplotlib.pyplot as plt


# Prepare data
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

# Design model using class
class LogisticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.modules.linear.Linear(1,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred

model = LogisticModel()

# Construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# Training cycle
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('Epoch:', epoch, 'loss=', loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Print results
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

# Test
y_test = model(torch.Tensor([[4.0]]))
print('y_pred = ', y_test.item())

# plt
x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200,1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('h')
plt.ylabel('pass')
plt.legend()
plt.grid()
plt.show()