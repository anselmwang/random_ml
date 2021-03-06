N_SYMBOL = 26
N_HID = 60
N_EPOCH = 5000

def my_sum(a, b):
    return (a + b) % N_SYMBOL

import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense_a = nn.Embedding(N_SYMBOL, N_HID)
        self.dense_b = nn.Embedding(N_SYMBOL, N_HID)
        self.composition = nn.Linear(N_HID + N_HID, N_HID)
        self.output = nn.Linear(N_HID, 1)

    def forward(self, a, b):
        final_input = torch.relu(self.composition(torch.cat( (torch.relu(self.dense_a(a)),
                                     torch.relu(self.dense_b(b))),
                                    1)))

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
                                                                                                     test_size=0.2,
                                                                                                     random_state=0,
                                                                                                     shuffle=True)

t_input_a = torch.LongTensor(input_a)
t_input_a_test = torch.LongTensor(input_a_test)
t_input_b = torch.LongTensor(input_b)
t_input_b_test = torch.LongTensor(input_b_test)
t_output = torch.FloatTensor(output).view(-1, 1)
t_output_test = torch.FloatTensor(output_test).view(-1, 1)


criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

loss_history = []
test_loss_history = []
for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input_a, t_input_b), t_output)
    loss_test = criterion(net(t_input_a_test, t_input_b_test), t_output_test)
    loss.backward()
    optimizer.step()
    print(i, loss.item(), loss_test.item())
    loss_history.append(loss.item())
    test_loss_history.append(loss_test.item())

t_pred_output_test = net(t_input_a_test, t_input_b_test)
for i in range(t_pred_output_test.shape[0])[:10]:
    print(input_a_test[i], input_b_test[i], t_pred_output_test[i].item(), my_sum(input_a_test[i], input_b_test[i]))


import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(N_EPOCH), loss_history, label="loss")
plt.plot(np.arange(N_EPOCH), test_loss_history, label="test_loss")
plt.legend()
plt.show()



