import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import time

N_EPOCH = 2000
MAX_LEN = 3
MAX_NUM = 999
TEST_SIZE = 0.5
#BATCH_SIZE = int((MAX_NUM + 1) * TEST_SIZE)
BATCH_SIZE = 50
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
#torch.cuda.manual_seed_all(SEED)

class MyLSTM(nn.Module):
    def __init__(self, n_token, token_padding_idx, n_emb_dim, n_lstm_dim):
        super(MyLSTM, self).__init__()
        self.emb = nn.Embedding(n_token, n_emb_dim, padding_idx=token_padding_idx)
        self.lstm = nn.LSTM(input_size=n_emb_dim,
                            hidden_size=n_lstm_dim,
                            num_layers=1,
                            batch_first=True)
        self.output = nn.Linear(in_features=n_lstm_dim,
                                out_features=1)

        self.n_lstm_dim = n_lstm_dim

    def _init_hidden(self, batch_size):
        hidden_a = torch.randn(1, batch_size, self.n_lstm_dim)
        hidden_b = torch.randn(1, batch_size, self.n_lstm_dim)
        return hidden_a, hidden_b

    def forward(self, x, x_lengths, batch_size):
        """ x.shape (batch_size, max_seq_len)"""

        """(batch_size, max_seq_len, n_emb_dim)"""
        x = self.emb(x)

        """(batch_size, max_seq_len, n_lstm_dim)"""
        # TODO: The pack & pad makes the last state not correct, so with these 2 lines, len <3 number are not fit well.
        #x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        x, (h_n, c_n) = self.lstm(x, self._init_hidden(batch_size))
        #x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        """(batch_size, n_lstm_dim)"""
        x = h_n.contiguous().view(-1, self.n_lstm_dim)

        """(batch_size, 1)"""
        x = self.output(x)

        return x.view(batch_size)


x = []
y = []
x_lengths = []

for i in range(MAX_NUM, -1, -1):
    x.append([int(c) + 1 for c in str(i)] + [0] * (MAX_LEN-len(str(i))))
    x_lengths.append(len(str(i)))
    y.append(i)
x = np.asarray(x)
y = np.asarray(y)
x_lengths = np.asarray(x_lengths)
import sklearn.model_selection as model_selection

x, x_test, y, y_test, x_lengths, x_lengths_test = model_selection.train_test_split(x,
                                                        y,
                                                        x_lengths,
                                                        test_size=TEST_SIZE,
                                                        random_state=0,
                                                        shuffle=True)

def order_by_length(x, y, x_lengths):
    x, y, x_lengths = x[np.argsort(x_lengths)[::-1]], y[np.argsort(x_lengths)[::-1]], x_lengths[np.argsort(x_lengths)[::-1]]
    return x, y, x_lengths

# x, y, x_lengths = order_by_length(x, y, x_lengths)
# t_x = torch.LongTensor(x)
# t_y = torch.FloatTensor(y)

x_test, y_test, x_lengths_test = order_by_length(x_test, y_test, x_lengths_test)
t_x_test = torch.LongTensor(x_test)
t_y_test = torch.FloatTensor(y_test)

net = MyLSTM(n_token=10+1,
              token_padding_idx=0,
              n_emb_dim=20,
              n_lstm_dim=20)

optimizer = optim.Adam(net.parameters())

loss_history = []
loss_test_history = []

start_time = time.time()
for epoch_no in range(N_EPOCH):
    net.zero_grad()
    # for batch_no in range(int(len(t_x)/BATCH_SIZE)):
    #     loss = F.mse_loss(net(t_x[batch_no*BATCH_SIZE: (batch_no+1)*BATCH_SIZE],
    #                           x_lengths[batch_no*BATCH_SIZE: (batch_no+1)*BATCH_SIZE],
    #                           BATCH_SIZE),
    #                       t_y[batch_no*BATCH_SIZE: (batch_no+1)*BATCH_SIZE])
    #     loss.backward()
    #     optimizer.step()

    for batch_no in range(int(len(x)/BATCH_SIZE)):
        x_batch = x[batch_no * BATCH_SIZE: (batch_no + 1) * BATCH_SIZE]
        y_batch = y[batch_no * BATCH_SIZE: (batch_no + 1) * BATCH_SIZE]
        x_lengths_batch = x_lengths[batch_no * BATCH_SIZE: (batch_no + 1) * BATCH_SIZE]
        x_batch, y_batch, x_lengths_batch = order_by_length(x_batch, y_batch, x_lengths_batch)
        t_x_batch = torch.LongTensor(x_batch)
        t_y_batch = torch.FloatTensor(y_batch)

        loss = F.mse_loss(net(t_x_batch,
                              x_lengths_batch,
                              BATCH_SIZE),
                          t_y_batch)
        loss.backward()
        optimizer.step()

    loss_test = F.mse_loss(net(t_x_test, x_lengths_test, len(t_x_test)), t_y_test)
    print(epoch_no, loss.item(), loss_test.item())
    loss_history.append(loss.item())
    loss_test_history.append(loss_test.item())

print(time.time()-start_time)

import matplotlib.pyplot as plt
pred_y = net(t_x_test, x_lengths_test, len(t_x_test))
plt.scatter(t_y_test, pred_y.detach())
plt.show()