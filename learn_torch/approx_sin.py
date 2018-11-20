import numpy as np

def f(x):
    return np.sin(x)

import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense0 = nn.Linear(1, 10)
        self.dense1 = nn.Linear(10, 1)

    def forward(self, x):
        return self.dense1(torch.relu(self.dense0(x)))

net = Net()

input = torch.tensor(np.linspace(-5, 5, 50), dtype=torch.float32).view(-1, 1)
output = f(input)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(1000):
    net.zero_grad()
    loss = criterion(net(input), output)
    loss.backward()
    optimizer.step()
    print(epoch, loss.item())

import matplotlib.pyplot as plt
plt.matshow(torch.relu(net.dense0(input)))
plt.show()