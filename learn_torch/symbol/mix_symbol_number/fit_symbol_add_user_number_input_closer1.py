def my_sum(a, b):
    return (a + b) % 10

import torch
import torch.nn as nn
import torch.optim as optim

N_SYMBOL = 10
N_HID = 8
N_EPOCH = 10000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense = nn.Linear(2, N_HID)
        self.output = nn.Linear(N_HID, 1)

    def forward(self, x):
        final_input = torch.relu(self.dense(x))
        return self.output(final_input)


net = Net()

input_a = []
input_b = []
output = []
for i in range(N_SYMBOL):
    for j in range(N_SYMBOL):
        input_a.append(i)
        input_b.append(j)
        output.append(my_sum(i, j))
# to numeric
import sklearn.model_selection as model_selection

input_a, input_a_test, input_b, input_b_test, output, output_test = model_selection.train_test_split(input_a,
                                                                                                     input_b,
                                                                                                     output,
                                                                                                     test_size=0.75,
                                                                                                     random_state=0,
                                                                                                     shuffle=True)

def one_hot(input):
    return torch.FloatTensor(input).view(-1, 1)

t_input_a = one_hot(input_a)
t_input_a_test = one_hot(input_a_test)
t_input_b = one_hot(input_b)
t_input_b_test = one_hot(input_b_test)

t_input = torch.cat((t_input_a, t_input_b), 1)
t_input_test = torch.cat((t_input_a_test, t_input_b_test), 1)
t_output = torch.FloatTensor(output).view(-1, 1)
t_output_test = torch.FloatTensor(output_test).view(-1, 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input), t_output)
    loss_test = criterion(net(t_input_test), t_output_test)

    loss.backward()
    optimizer.step()
    print(i, loss.item(), loss_test.item())

