import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLSTM(nn.Module):
    def __init__(self, n_token, token_padding_idx, n_emb_dim, n_lstm_layer, n_lstm_hidden, n_tag, tag_padding_idx, batch_size):
        super(MyLSTM, self).__init__()
        self.emb = nn.Embedding(n_token, n_emb_dim, padding_idx=token_padding_idx)
        self.lstm = nn.LSTM(input_size=n_emb_dim, hidden_size=n_lstm_hidden, num_layers=n_lstm_layer, batch_first=True)
        self.output = nn.Linear(n_lstm_hidden, n_tag)

        self.n_lstm_layer = n_lstm_layer
        self.batch_size = batch_size
        self.n_lstm_hidden = n_lstm_hidden
        self.n_tag = n_tag
        self.tag_padding_idx = tag_padding_idx

    def _init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.n_lstm_layer, self.batch_size, self.n_lstm_hidden)
        hidden_b = torch.randn(self.n_lstm_layer, self.batch_size, self.n_lstm_hidden)

        return (hidden_a, hidden_b)

    def forward(self, x, x_lengths):
        """ x.shape is (batch_size, seq_len)"""
        x = self.emb(x)

        x = nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        hidden_init = self._init_hidden()

        """shape (batch_size, seq_len, n_lstm_hidden)"""
        x, _ = self.lstm(x, hidden_init)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        """shape (batch_size * seq_len, n_lstm_hidden)"""
        x = x.contiguous().view(-1, self.n_lstm_hidden)

        """ (batch_size*seq_len, n_tag)"""
        x = self.output(x)
        x = F.log_softmax(x, dim=1)

        return x.view(self.batch_size, -1, self.n_tag)

    def loss(self, y_pred, y):
        """y.shape (batch_size, max_seq_len)
           y_pred.shape (batch_size, max_seq_len, n_tag)
        """

        y_pred = y_pred.view(-1, self.n_tag)
        y = y.view(-1)
        return F.nll_loss(y_pred, y, ignore_index=self.tag_padding_idx)

X = [['is', 'it', 'too', 'late', 'now', 'say', 'sorry'],
     ['ooh', 'ooh'],
     ['sorry', 'yeah']
     ]
Y = [['VB', 'PRP', 'RB', 'RB', 'RB', 'VB', 'JJ'],
     [ 'NNP', 'NNP'],
     ['JJ', 'NNP']]

vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah': 9}
tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}

X = [[vocab[word] for word in seq] for seq in X]
Y = [[tags[tag] for tag in seq] for seq in Y]
x_lengths = [len(seq) for seq in X]
t_X = torch.zeros(len(X), max(x_lengths), dtype=torch.long)
t_Y = torch.zeros(len(Y), max(x_lengths), dtype=torch.long)
for no, (seq_x, seq_y) in enumerate(zip(X, Y)):
    t_X[no, :len(seq_x)] = torch.LongTensor(seq_x)
    t_Y[no, :len(seq_y)] = torch.LongTensor(seq_y)


lstm = MyLSTM(n_token=len(vocab), token_padding_idx=vocab['<PAD>'], n_emb_dim=10, n_lstm_layer=1,
              n_lstm_hidden=10, n_tag=len(tags), tag_padding_idx=tags["<PAD>"], batch_size=3)
y_pred = lstm.forward(t_X, x_lengths)
loss = lstm.loss(y_pred, t_Y)