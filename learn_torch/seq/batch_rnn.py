import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

"""
Blog post:
Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health:
https://medium.com/@_willfalcon/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""


class BieberLSTM(nn.Module):
    def __init__(self, nb_layers, nb_lstm_units=100, embedding_dim=3, batch_size=3):
        super(BieberLSTM, self).__init__()
        self.vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8,
                      'yeah': 9}
        self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}

        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.nb_tags = len(self.tags)

        # build actual NN
        self.__build_model()

    def __build_model(self):
        # build embedding layer first
        nb_vocab_words = len(self.vocab)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.vocab['<PAD>']
        self.word_embedding = nn.Embedding(
            num_embeddings=nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_layers,
            batch_first=True,
        )

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_layers, self.batch_size, self.nb_lstm_units)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        #batch_size, seq_len, _ = X.shape
        batch_size, seq_len = X.shape

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        X, _ = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden_to_tag(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size, seq_len, self.nb_tags)

        Y_hat = X
        return Y_hat


sent_1_x = ['is', 'it', 'too', 'late', 'now', 'say', 'sorry']
sent_1_y = ['VB', 'PRP', 'RB', 'RB', 'RB', 'VB', 'JJ']
sent_2_x = ['ooh', 'ooh']
sent_2_y = ['NNP', 'NNP']
sent_3_x = ['sorry', 'yeah']
sent_3_y = ['JJ', 'NNP']
X = [sent_1_x, sent_2_x, sent_3_x]
Y = [sent_1_y, sent_2_y, sent_3_y]

# map sentences to vocab
vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah': 9}
# fancy nested list comprehension
X =  [[vocab[word] for word in sentence] for sentence in X]
# X now looks like:
# [[1, 2, 3, 4, 5, 6, 7], [8, 8], [7, 9]]

tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
# fancy nested list comprehension
Y =  [[tags[tag] for tag in sentence] for sentence in Y]
# Y now looks like:
# [[1, 2, 3, 3, 3, 1, 4], [5, 5], [4, 5]]

# get the length of each sentence
X_lengths = [len(sentence) for sentence in X]
# create an empty matrix with padding tokens
pad_token = vocab['<PAD>']
longest_sent = max(X_lengths)
batch_size = len(X)
padded_X = np.ones((batch_size, longest_sent)) * pad_token
# copy over the actual sequences
for i, x_len in enumerate(X_lengths):
  sequence = X[i]
  padded_X[i, 0:x_len] = sequence[:x_len]

# get the length of each sentence
Y_lengths = [len(sentence) for sentence in Y]
# create an empty matrix with padding tokens
pad_token = tags['<PAD>']
longest_sent = max(Y_lengths)
batch_size = len(Y)
padded_Y = np.ones((batch_size, longest_sent)) * pad_token
# copy over the actual sequences
for i, y_len in enumerate(Y_lengths):
  sequence = Y[i]
  padded_Y[i, 0:y_len] = sequence[:y_len]

lstm = BieberLSTM(1)

criterion = nn.NLLLoss(ignore_index=tags['<PAD>'])

Y_hat = lstm(torch.LongTensor(padded_X), X_lengths)

print(criterion(Y_hat.view(-1, lstm.nb_tags), torch.LongTensor(padded_Y).view(-1)))
