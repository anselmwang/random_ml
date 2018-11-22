import numpy as np
def f(x, y, z):
    return x ** 2 + (y+10) ** 2

import torch
import torch.nn as nn
import torch.optim as optim

N_HID = 40
N_EPOCH = 12000
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gate0 = nn.Linear(3, 3)
        self.gate1 = nn.Linear(3, 3)
        self.dense0 = nn.Linear(2, N_HID)
        self.dense1 = nn.Linear(N_HID, 1)

    def forward(self, x):
        hid0 = (torch.softmax(self.gate0(x), dim=1) * x).sum(1).view(-1, 1)
        hid1 = (torch.softmax(self.gate1(x), dim=1) * x).sum(1).view(-1, 1)
        return self.dense1(torch.relu(self.dense0(torch.cat((hid0, hid1), 1))))

net = Net()
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
z = np.linspace(-5, 5, 20)
x, y, z = np.meshgrid(x, y, z)
output = f(x, y, z)

t_input = torch.tensor(np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1))), dtype=torch.float32)
t_output = torch.tensor(output.reshape(-1, 1), dtype=torch.float32)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input), t_output)
    loss.backward()
    optimizer.step()
    print(i, loss.item())

import seaborn as sns
import matplotlib.pyplot as plt

mask0 = torch.softmax(net.gate0(t_input), dim=1)
mask1 = torch.softmax(net.gate1(t_input), dim=1)
f, axes = plt.subplots(2, 3)
sns.distplot(mask0[:, 0].detach().numpy(), ax=axes[0, 0])
sns.distplot(mask0[:, 1].detach().numpy(), ax=axes[0, 1])
sns.distplot(mask0[:, 2].detach().numpy(), ax=axes[0, 2])
sns.distplot(mask1[:, 0].detach().numpy(), ax=axes[1, 0])
sns.distplot(mask1[:, 1].detach().numpy(), ax=axes[1, 1])
sns.distplot(mask1[:, 2].detach().numpy(), ax=axes[1, 2])
plt.show()