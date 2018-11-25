import numpy as np
def f(x, y):
    return (x + y) % 26

import torch
import torch.nn as nn
import torch.optim as optim

N_HID = 40
N_EPOCH = 6000
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense0 = nn.Linear(2, N_HID)
        self.dense1 = nn.Linear(N_HID, 1)

    def forward(self, x):
        return  self.dense1(torch.relu(self.dense0(x)))

net = Net()

N_POINT = 50
# x = np.linspace(-5, 5, 50)
# y = np.linspace(-5, 5, 50)
x = np.linspace(0, 25, N_POINT)
y = np.linspace(0, 25, N_POINT)
x, y = np.meshgrid(x, y)
output = f(x, y)

t_input = torch.tensor(np.hstack((x.reshape(-1, 1), y.reshape(-1, 1))), dtype=torch.float32)
t_output = torch.tensor(output.reshape(-1, 1), dtype=torch.float32)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input), t_output)
    loss.backward()
    optimizer.step()
    print(i, loss.item())

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, output, label="orig")
# ax.plot_surface(x, y, net(t_input).detach().numpy().reshape(N_POINT, N_POINT), label="fit")

#ax.plot_wireframe(x, y, output, color="red", label="orig")
ax.plot_wireframe(x, y, net(t_input).detach().numpy().reshape(N_POINT, N_POINT), color="green", label="fit")

#plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
