from approx_sin_common import *

import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import skorch
import skorch.net

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        #self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NNModel:
    def __init__(self, net):
        self._net = net

        #self._optimizer = optim.SGD(net.parameters(), lr=0.001)
        #self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, 'min', factor=0.5)
        #self._optimizer = optim.SGD(net.parameters(), lr=0.001)
        self._optimizer = optim.Adagrad(net.parameters(), lr=0.5)
        self._criterion = nn.MSELoss()

        self.train_loss_list = []
        self.test_loss_list = []

    def _forward(self, input_mat):
        input_var = Variable(torch.Tensor(input_mat))
        output = self._net(input_var)
        return output

    def get_loss(self, input_mat, y):
        output = self._forward(input_mat)
        loss = self._criterion(output, Variable(torch.Tensor(y)))
        return loss, loss.data.numpy()

    def fit(self, train_mat, train_y):
        global test_mat, test_y
        data_set = data.TensorDataset(torch.FloatTensor(train_mat),
                                      torch.FloatTensor(train_y))
        dataloader = data.DataLoader(data_set, batch_size=4,
                                     shuffle=True)
        for epoch_no in range(2000):
            print("Epoch %s" % epoch_no)
            train_loss_num = 0
            for iter, train_batch in enumerate(dataloader):
                output = self._net(Variable(train_batch[0]))
                loss = self._criterion(output, Variable(train_batch[1].type(torch.FloatTensor)))
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()  # Does the update

                loss_num = loss.data.numpy()
                train_loss_num += loss_num

            train_loss_num /= iter + 1
            self.train_loss_list.append(train_loss_num)
            print("train_loss:%.4f" % train_loss_num)

            loss, loss_num = self.get_loss(test_mat, test_y)
            self.test_loss_list.append(loss_num)
            print("test_loss:%.4f" % loss_num)

            if hasattr(self, "_scheduler"):
                if isinstance(self._scheduler.best, float) and self._scheduler.best == float("inf"):
                    self._scheduler.best = loss_num[0]
                else:
                    self._scheduler.step(loss_num[0])
                print("lr: %.8f" % self._optimizer.param_groups[0]['lr'])

    def predict(self, input_mat):
        output = self._forward(input_mat)
        print(output.data.numpy().shape)
        return output.data.numpy()


plt.plot(test_x, test_y, label="original")
eval(ConstantModel(0.), "Const 0")
# nn_model = NNModel(Net())
# eval(nn_model, "NN")

start = time.clock()
nn_model = skorch.net.NeuralNetRegressor(
    Net,
    optimizer=optim.Adagrad,
    max_epochs=2000,
    lr=0.5,
    batch_size=4,
)
eval(nn_model, "NN")
print(time.clock()-start)
plt.legend()


# plt.figure()
# train_loss_list = nn_model.train_loss_list
# test_loss_list = nn_model.test_loss_list
# plt.plot(range(len(train_loss_list)), train_loss_list, label="train loss")
# plt.plot(range(len(test_loss_list)), test_loss_list, label="test loss")
# plt.legend()


## nn_model._optimizer.param_groups[0]['lr'] = nn_model._optimizer.param_groups[0]['lr'] * 0.5
# plt.figure()
# eval(nn_model, "NN")
# plt.legend()
