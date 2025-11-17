import torch


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_1 = torch.tensor([1.0])
w_2 = torch.tensor([1.0])
w_1.requires_grad = True
w_2.requires_grad = True
b = torch.tensor([1.0])
b.requires_grad = True

def forward(x):
    return w_1 * (x) ** 2 + w_2 * x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('Predict (before training):', 4, forward(4.0).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        loss_val.backward()
        print('\tgrad:', x, y, w_1.grad.item(), w_2.grad.item(), b.grad.item())
        w_1.data = w_1.data - 0.01 * w_1.grad.data
        w_2.data = w_2.data - 0.01 * w_2.grad.data
        b.data = b.data - 0.01 * b.grad.data
        w_1.grad.data.zero_()
        w_2.grad.data.zero_()
        b.grad.data.zero_()
    print('Epoch:', epoch, 'loss=', loss_val.item())

print('Predict (before training):', 4, forward(4.0).item())
