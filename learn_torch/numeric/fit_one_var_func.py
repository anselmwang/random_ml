import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})

def f(x):
    #y = np.log(x)
    #y[np.isnan(y)] = -1.
    #y = (x > 0).astype(float)
    y = np.sin(x)
    return y

import torch
import torch.nn as nn
import torch.optim as optim

N_HID = 30
N_EPOCH = 2000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense0 = nn.Linear(1, N_HID)
        self.dense1 = nn.Linear(N_HID, 1)

    def forward(self, x):
        return self.dense1(torch.relu(self.dense0(x)))

net = Net()

x = np.linspace(-5, 5, 5)
output = f(x)

t_input = torch.tensor(x, dtype=torch.float32).view(-1, 1)
t_output = torch.tensor(output, dtype=torch.float32).view(-1, 1)


criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

plt.plot(x, f(x), label="origin")

for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input), t_output)
    loss.backward()
    optimizer.step()
    if i % int(N_EPOCH / 5) == 0:
        plt.plot(x, net(t_input).detach().view(-1).numpy(), "--", label="fit %s" % i)
    print(i, loss.item())

plt.plot(x, net(t_input).detach().view(-1).numpy(), label="final")

plt.legend()
plt.show()

