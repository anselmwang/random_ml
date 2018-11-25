def my_sum(a, b):
    return (a+b) % 10

import torch
import torch.nn as nn
import torch.optim as optim

N_SYMBOL = 10
N_HID = 8
N_EPOCH = 10000
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense_a = nn.Linear(N_SYMBOL, N_HID)
        self.dense_b = nn.Linear(N_SYMBOL, N_HID)
        # self.transform = nn.Linear(N_HID + N_HID, N_HID + N_HID)
        # self.output = nn.Linear(N_HID + N_HID, N_SYMBOL)
        # self.output = nn.Linear(N_HID + N_HID, 1)

        self.output = nn.Linear(N_HID, 1)

    def forward(self, a, b):
        # a_out = torch.relu(self.dense_a(a))
        # b_out = torch.relu(self.dense_b(b))
        # final_input = torch.cat((a_out, b_out), 1)
        # final_input = torch.relu(self.transform(final_input))
        # return self.output(final_input)
        #return torch.log_softmax(self.output(final_input), dim=1)

        final_input = torch.relu(self.dense_a(a) + self.dense_b(b))
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
    t_input = torch.zeros(len(input), N_SYMBOL, dtype=torch.float32)
    t_input.scatter_(1, torch.tensor(input, dtype=torch.long).reshape(-1, 1), 1.)
    return t_input

t_input_a = one_hot(input_a)
t_input_a_test = one_hot(input_a_test)
t_input_b = one_hot(input_b)
t_input_b_test = one_hot(input_b_test)
t_output = torch.FloatTensor(output).view(-1, 1)
t_output_test = torch.FloatTensor(output_test).view(-1, 1)
# t_output = torch.LongTensor(output)
# t_output_test = torch.LongTensor(output_test)


#criterion = nn.NLLLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input_a, t_input_b), t_output)
    loss_test = criterion(net(t_input_a_test, t_input_b_test), t_output_test)

    loss.backward()
    optimizer.step()
    print(i, loss.item(), loss_test.item())

torch.argmax(net(one_hot([5]), one_hot([8])), dim=1)
#
#
# torch.argmax(net(t_input_a_test, t_input_b_test), dim=1)
#
# for i, (a, b) in enumerate(zip(input_a, input_b)):
#     if a == 9 and b == 8:
#         print(i)
