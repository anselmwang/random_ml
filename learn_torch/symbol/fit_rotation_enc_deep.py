def rotation_encode(c):
    return chr(ord('a') +((ord(c) - ord('a') + 5) % 26))

import torch
import torch.nn as nn
import torch.optim as optim

N_SYMBOL = 26
N_HID = 5
N_TRANSFORM_LAYER = 0
N_EPOCH = 500
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.emb = nn.Embedding(N_SYMBOL, N_HID)
        self.transform_layers = []
        for layer_no in range(N_TRANSFORM_LAYER):
            self.transform_layers.append(nn.Linear(N_HID, N_HID))
        self.output = nn.Linear(N_HID, N_SYMBOL)

    def forward(self, x):
        out = self.emb(x)
        for layer_no in range(N_TRANSFORM_LAYER):
            out = torch.relu(self.transform_layers[layer_no](out))
        return torch.log_softmax(self.output(out), 1)
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
t_input = torch.tensor(input, dtype=torch.long)
t_output = torch.tensor(output, dtype=torch.long)

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters())


for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input), t_output)
    acc = (torch.argmax(net(t_input), dim=1) == t_output).sum().item() / float(t_output.shape[0])
    loss.backward()
    optimizer.step()
    print(i, loss.item(), acc)

test_c = 'd'
print( chr(torch.argmax(net(torch.tensor([ord(test_c) - ord('a')], dtype=torch.long)), dim=1) + ord('a')),
       rotation_encode(test_c))

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
plt.imshow(metrics.pairwise.cosine_similarity(net.emb.weight.detach().numpy()))
plt.colorbar()
plt.show()