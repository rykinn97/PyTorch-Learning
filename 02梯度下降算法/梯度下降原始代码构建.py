import numpy as np
import matplotlib.pyplot as plt 


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * (w * x - y) * x
    return grad / len(xs)

cost_list = []
print('Predict (before training):', 4, forward(4.0))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    cost_list.append(cost_val)
    grad_val = gradient(x_data, y_data)
    a = 0.01
    w -= a * grad_val
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
print('Predict (after training):', 4, forward(4.0))

plt.plot(range(100), cost_list)
plt.xlabel('Epoch')
plt.ylabel('cost')
plt.grid()
plt.show()

