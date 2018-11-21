import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})

def f(x):
    # y = np.log(x)
    # y[np.isnan(y)] = -1.
    y = (x > 0).astype(float)
    return y

import torch
import torch.nn as nn
import torch.optim as optim

N_HID = 10
N_EPOCH = 6000
N_PIECE = 2
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gate0 = nn.Linear(1, N_HID)
        self.gate1 = nn.Linear(N_HID, N_PIECE)
        #self.gate = nn.Linear(1, N_PIECE)
        self.dense0 = nn.Linear(1, N_HID)
        self.dense1 = nn.Linear(N_HID, N_PIECE)

    def forward(self, x):
        mask = torch.softmax(self.gate1(torch.relu(self.gate0(x))), dim=1)
        #mask = torch.softmax(self.gate(x), dim=1)
        return (self.dense1(torch.relu(self.dense0(x))) * mask).sum(1).reshape(-1, 1)

net = Net()

x = np.linspace(-5, 5, 50)
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


plt.subplot(131)
#mask = torch.softmax(net.gate(t_input), dim=1)
mask = torch.softmax(net.gate1(torch.relu(net.gate0(t_input))), dim=1)
for i in range(N_PIECE):
    plt.plot(t_input.view(-1).detach().numpy(), mask[:, i].detach().numpy(), label="mask %s" % i)
plt.legend()
plt.subplot(132)
result = net.dense1(torch.relu(net.dense0(t_input)))
for i in range(N_PIECE):
    plt.plot(t_input.view(-1).detach().numpy(), result[:, i].detach().numpy(), label="result %s" % i)
plt.legend()
plt.subplot(133)
plt.plot(x, (result*mask).sum(1).detach().numpy(), label="final")
