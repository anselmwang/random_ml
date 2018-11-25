def my_sum(a, b):
    return (a+b) % 10

import torch
import torch.nn as nn
import torch.optim as optim

N_SYMBOL = 10
N_HID = 10
N_EPOCH = 10000
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense_a = nn.Linear(N_SYMBOL, N_HID)
        self.dense_b = nn.Linear(N_SYMBOL, N_HID)
        self.output = nn.Linear(N_HID + N_HID, 1)

    def forward(self, a, b):
        a_out = torch.relu(self.dense_a(a))
        b_out = torch.relu(self.dense_b(b))
        final_input = torch.cat((a_out, b_out), 1)
        return self.output(final_input)
net = Net()

input_a = []
input_b = []
output = []
for i in range(N_SYMBOL):
    for j in range(N_SYMBOL):
        input_a.append(i)
        input_b.append(j)
        output.append(my_sum(i, i))
# to numeric

import sklearn.model_selection as model_selection
input_a, input_a_test, input_b, input_b_test, output, output_test = model_selection.train_test_split(input_a,
                                                                                                             input_b,
                                                                                                             output,
                                                                                                             test_size=0.5,
                                                                                                             random_state=0,
                                                                                                             shuffle=True)
def one_hot(input):
    t_input = torch.zeros(len(input), N_SYMBOL, dtype=torch.float32)
    t_input.scatter_(1, torch.tensor(input, dtype=torch.long).reshape(-1, 1), 1.)
    return t_input

t_input_a = one_hot(input_a)
t_input_a_test = one_hot(input_a_test)
t_input_b = one_hot(input_b)
t_input_b_test = one_hot(input_b_test)
t_output = torch.FloatTensor(output).view(-1, 1)
t_output_test = torch.FloatTensor(output_test).view(-1, 1)


criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input_a, t_input_b), t_output)
    loss_test = criterion(net(t_input_a_test, t_input_b_test), t_output_test)

    loss.backward()
    optimizer.step()
    print(i, loss.item(), loss_test.item())