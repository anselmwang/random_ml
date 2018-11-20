import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense = nn.Linear(2, 1)

    def forward(self, x):
        return self.dense(x)

fixed_net = Net()
fixed_net.dense.weight.data = torch.Tensor([[1., 1.]])
fixed_net.dense.bias.data = torch.Tensor([1.])
print(fixed_net(torch.tensor([2., 4.])))

import torch.optim as optim
learnable_net = Net()
print(learnable_net(torch.tensor([2., 4.])))

input = torch.rand(10, 2)
target = fixed_net(input).detach()

optimizer = optim.Adam(learnable_net.parameters(), lr=1)
criteria = nn.MSELoss()

for epoch in range(30):
    optimizer.zero_grad()  # zero the gradient buffers
    output = learnable_net(input)
    loss = criteria(output, target)
    loss.backward()
    optimizer.step()
    print("====epoch %s, loss %.4f====" % (epoch, loss.item()))
    # for name, param in learnable_net.named_parameters():
    #     print(name, param)

