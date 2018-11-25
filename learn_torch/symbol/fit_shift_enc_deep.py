def shift_encode(c, delta):
    idx = ord(c) - ord('a')
    new_idx = (idx + delta) % 26
    return chr(ord('a') + new_idx)

import torch
import torch.nn as nn
import torch.optim as optim

N_SYMBOL = 26
N_EMB_DIM = 10
N_EPOCH = 1000
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.emb = nn.Embedding(N_SYMBOL, N_EMB_DIM)
        self.output = nn.Linear(N_EMB_DIM+1, N_SYMBOL)

    def forward(self, c_idx, delta):
        emb_out = self.emb(c_idx)
        final_input = torch.cat((emb_out, delta.view(-1, 1)), 1)
        return torch.log_softmax(self.output(final_input), 1)
net = Net()

input_c = []
input_delta = []
output = []
for i in range(26):
    c = chr(i + ord('a'))
    for delta in range(26):
        input_c.append(c)
        input_delta.append(delta)
        output.append(shift_encode(c, delta))
# to numeric
input_c = [ord(c) - ord('a') for c in input_c]
output = [ord(c) - ord('a') for c in output]
t_input_c = torch.LongTensor(input_c)
t_input_delta = torch.FloatTensor(input_delta)
t_output = torch.LongTensor(output)

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters())

for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input_c, t_input_delta), t_output)
    acc = (torch.argmax(net(t_input_c, t_input_delta), dim=1) == t_output).sum().item() / float(t_output.shape[0])
    loss.backward()
    optimizer.step()
    print(i, loss.item(), acc)

test_c = 'd'
test_delta = 5
print( chr(torch.argmax(net(torch.LongTensor([ord(test_c) - ord('a')]), torch.FloatTensor([test_delta])), dim=1) + ord('a')),
       shift_encode(test_c, test_delta))
