def rotation_encode(c):
    return chr(ord('a') +((ord(c) - ord('a') + 5) % 26))

import torch
import torch.nn as nn
import torch.optim as optim

N_SYMBOL = 26
N_EPOCH = 100
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense = nn.Linear(N_SYMBOL, N_SYMBOL)

    def forward(self, x):
        return torch.log_softmax(self.dense(x), 1)
net = Net()

input = []
output = []
for i in range(26):
    c = chr(i+ord('a'))
    input.append(c)
    output.append(rotation_encode(c))
# to numeric
input = [ord(c) - ord('a') for c in input]
output = [ord(c) - ord('a') for c in output]
t_input = torch.zeros(len(input), N_SYMBOL, dtype=torch.float32)
t_input.scatter_(1, torch.tensor(input, dtype=torch.long).reshape(-1, 1), 1.)
t_output = torch.tensor(output, dtype=torch.long)

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters())


for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input), t_output)
    acc = (torch.argmax(net(t_input), dim=1) == t_output).sum().item() / t_output.shape[0]
    loss.backward()
    optimizer.step()
    print(i, loss.item(), acc)


