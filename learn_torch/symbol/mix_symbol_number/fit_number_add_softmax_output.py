N_SYMBOL = 26
N_HID = 60
N_EPOCH = 40000

def my_sum(a, b):
    return (a + b) % N_SYMBOL

import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense_a = nn.Linear(1, N_HID)
        self.dense_b = nn.Linear(1, N_HID)
        self.output = nn.Linear(N_HID, N_SYMBOL)

    def forward(self, a, b):
        final_input = torch.relu(self.dense_a(a) + self.dense_b(b))
        return torch.log_softmax(self.output(final_input), 1)


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
                                                                                                     test_size=0.5,
                                                                                                     random_state=0,
                                                                                                     shuffle=True)


# def one_hot(input):
#     t_input = torch.zeros(len(input), N_SYMBOL, dtype=torch.float32)
#     t_input.scatter_(1, torch.tensor(input, dtype=torch.long).reshape(-1, 1), 1.)
#     return t_input

def one_hot(input):
    return torch.FloatTensor(input).view(-1, 1)

t_input_a = one_hot(input_a)
t_input_a_test = one_hot(input_a_test)
t_input_b = one_hot(input_b)
t_input_b_test = one_hot(input_b_test)
# t_output = torch.FloatTensor(output).view(-1, 1)
# t_output_test = torch.FloatTensor(output_test).view(-1, 1)
t_output = torch.LongTensor(output)
t_output_test = torch.LongTensor(output_test)



#criterion = nn.MSELoss()
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters())

loss_history = []
acc_history = []
mse_loss_history = []
test_loss_history = []
test_acc_history = []
test_mse_loss_history = []

for i in range(N_EPOCH):
    net.zero_grad()
    loss = criterion(net(t_input_a, t_input_b), t_output)
    loss_test = criterion(net(t_input_a_test, t_input_b_test), t_output_test)

    acc = (torch.argmax(net(t_input_a, t_input_b), 1) == t_output).sum().item() / float(len(t_output))
    acc_test = (torch.argmax(net(t_input_a_test, t_input_b_test), 1) == t_output_test).sum().item() / float(len(t_output_test))

    mse_loss = ((torch.argmax(net(t_input_a, t_input_b), 1) - t_output) ** 2).sum().item() / float(len(t_output))
    mse_loss_test = ((torch.argmax(net(t_input_a_test, t_input_b_test), 1) - t_output_test) ** 2).sum().item() / float(len(t_output_test))
    loss.backward()
    optimizer.step()
    print(i, loss.item(), acc, mse_loss, loss_test.item(), acc_test, mse_loss_test)
    loss_history.append(loss.item())
    acc_history.append(acc)
    mse_loss_history.append(mse_loss)
    test_loss_history.append(loss_test.item())
    test_acc_history.append(acc_test)
    test_mse_loss_history.append(mse_loss_test)

t_pred_output_test = torch.argmax(net(t_input_a_test, t_input_b_test), 1)
for i in range(t_pred_output_test.shape[0])[:10]:
    print(t_input_a_test[i], t_input_b_test[i], t_pred_output_test[i].item(), my_sum(t_input_a_test[i], t_input_b_test[i]))


import matplotlib.pyplot as plt
import numpy as np
plt.subplot(131)
plt.title("NLL Loss")
plt.plot(np.arange(N_EPOCH), loss_history, label="loss")
plt.plot(np.arange(N_EPOCH), test_loss_history, label="test_loss")
plt.legend()
plt.subplot(132)
plt.title("Acc")
plt.plot(np.arange(N_EPOCH), acc_history, label="loss")
plt.plot(np.arange(N_EPOCH), test_acc_history, label="test_loss")
plt.legend()
plt.subplot(133)
plt.title("MSE Loss")
plt.plot(np.arange(N_EPOCH), mse_loss_history, label="loss")
plt.plot(np.arange(N_EPOCH), test_mse_loss_history, label="test_loss")
plt.legend()
plt.show()



