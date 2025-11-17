import numpy as np 
import matplotlib.pyplot as plt 

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (w * x - y)

print('Predict (before training):', 4, forward(4.0))
for epoch in range(100):
    a = 0.01
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= a * grad
        print('\tgrad:', x, y, grad)
        loss_val = loss(x, y)
    print('Epoch:', epoch, 'w=', w, 'loss=', loss_val)
print('Predict (before training):', 4, forward(4.0))
