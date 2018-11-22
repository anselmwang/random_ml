import numpy as np
def f(x, y):
    return x ** 2

import torch
import torch.nn as nn
import torch.optim as optim

N_HID = 40
N_EPOCH = 6000
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gate = nn.Linear(2, 2)
        self.dense0 = nn.Linear(2, N_HID)
        self.dense1 = nn.Linear(N_HID, 1)

    def forward(self, x):
        x = torch.sigmoid(self.gate(x)) * x
        return self.dense1(torch.relu(self.dense0(x)))

net = Net()

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
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
import pandas as pd
import seaborn as sns
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(x, y, output, color="red", label="orig")
ax.plot_wireframe(x, y, net(t_input).detach().numpy().reshape(50, 50), color="green", label="fit")
#plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def to_np(t):
    return t.detach().numpy()

# investigate whether gate layer zero out y
mask = torch.sigmoid(net.gate(t_input))
h = sns.jointplot(to_np(mask[:, 0]), to_np(mask[:, 1]))
h.set_axis_labels('mask x', 'mask y')
plt.show()

plt.plot(to_np(t_input[:10, 1]), to_np(mask[:10, 1]))
plt.show()

out_mask = torch.sigmoid(net.gate(t_input)) * t_input
out_mask_x = out_mask[:, 0]
out_mask_y = out_mask[:, 1]
df = pd.DataFrame({"out_mask_x": to_np(out_mask_x).reshape(-1),
                   "out_mask_y": to_np(out_mask_y).reshape(-1)})
sns.violinplot(data=df)
plt.show()

# whether dense0 zero out y
sns.jointplot(to_np(net.dense0.weight[:, 0]), to_np(net.dense0.weight[:, 1]))
plt.show()

fake_mask = torch.zeros_like(t_input)
fake_mask[:, 0] = 1.
out0_x = torch.relu(net.dense0(t_input * fake_mask))
out1_x = net.dense1(torch.relu(net.dense0(t_input * fake_mask)))
fake_mask = torch.zeros_like(t_input)
fake_mask[:, 1] = 1.
out0_y = torch.relu(net.dense0(t_input * fake_mask))
out1_y = net.dense1(torch.relu(net.dense0(t_input * fake_mask)))

df = pd.DataFrame({"out0_x": out0_x.reshape(-1).detach().numpy(),
                   "out0_y": out0_y.reshape(-1).detach().numpy()})
sns.violinplot(data=df)
plt.show()

df = pd.DataFrame({"out1_x": out1_x.reshape(-1).detach().numpy(),
                   "out1_y": out1_y.reshape(-1).detach().numpy()})
sns.violinplot(data=df)
plt.show()



fig = plt.figure()
ax = fig.add_subplot(321, projection='3d')
fake_mask = torch.zeros_like(t_input)
fake_mask[:, 0] = 1.
part0 = net.dense1(torch.relu(net.dense0(t_input * fake_mask)))
ax.plot_wireframe(x, y, part0.detach().numpy().reshape(50, 50), color="green", label="fit")
plt.title("use x only")
plt.xlabel("x")
plt.ylabel("y")
ax = fig.add_subplot(322, projection='3d')
fake_mask = torch.zeros_like(t_input)
fake_mask[:, 1] = 1.
part1 = net.dense1(torch.relu(net.dense0(t_input * fake_mask)))
ax.plot_wireframe(x, y, part1.detach().numpy().reshape(50, 50), color="green", label="fit")
plt.title("use y only")
plt.xlabel("x")
plt.ylabel("y")
ax = fig.add_subplot(323, projection='3d')
ax.plot_wireframe(x, y, net(t_input).detach().numpy().reshape(50, 50), color="green", label="fit")
plt.title("fit")
plt.xlabel("x")
plt.ylabel("y")
ax = fig.add_subplot(325, projection='3d')
ax.plot_wireframe(x, y, mask[:,0].detach().numpy().reshape(50, 50), color="green", label="fit")
plt.title("calculated x weight")
plt.xlabel("x")
plt.ylabel("y")
ax = fig.add_subplot(326, projection='3d')
ax.plot_wireframe(x, y, mask[:,1].detach().numpy().reshape(50, 50), color="green", label="fit")
plt.title("calculated y weight")
plt.xlabel("x")
plt.ylabel("y")
plt.show()




