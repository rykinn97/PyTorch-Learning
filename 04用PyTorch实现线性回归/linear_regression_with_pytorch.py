import torch

# Prepare dataset
# 使用二维结构是PyTorch的标准用法,保持数据维度的一致性
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# Design model
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1) # 完整写法是：torch.nn.modules.linear.Linear

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
model = LinearModel()

# Construct loss and optimizer
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training cycle
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('Epoch:', epoch, 'loss=', loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Print results
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

# Test model
y_test = torch.Tensor([[4.0]])
print('y_pred = ', model(y_test).item())



