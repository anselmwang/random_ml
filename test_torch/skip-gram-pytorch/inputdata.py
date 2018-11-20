import collections
import os
import numpy as np
import collections

class Options(object):
  def __init__(self, datafile, vocabulary_size, dictionary=None):
    if dictionary is None:
      self.vocabulary_size = vocabulary_size
      self.vocabulary = self.read_data(datafile)
      data_or, self.count, self.vocab_words = self.build_dataset(self.vocabulary,
                                                                self.vocabulary_size)
      self.train_data = data_or
      self.save_path = "tmp"
      self.save_vocab()
    else:
      self.vocabulary = self.read_data(datafile)
      self.dictionary = dictionary
      data_or, self.count, self.vocab_words = self.read_dataset(self.vocabulary)
      self.train_data = data_or

    self.my_init_sampler()
    self.data_index = 0

  def my_init_sampler(self):
    count = [ele[1] for ele in self.count]
    pow_frequency = np.array(count) ** 0.75
    power = sum(pow_frequency)
    ratio = pow_frequency / power
    self.my_sample_table = np.arange(len(count))
    self.my_ratio = ratio

  def save_vocab(self):
    with open(os.path.join(self.save_path, "vocab.txt"), "w") as f:
      for i in range(len(self.count)):
        vocab_word = self.vocab_words[i]
        f.write("%s %d\n" % (vocab_word, self.count[i][1]))

  def read_dataset(self,words):
    count = collections.defaultdict(int)
    data = list()
    dictionary = self.dictionary
    for word in words:
      if word in dictionary:
        index = dictionary[word]
        count[word] += 1
      else:
        index = 0  # dictionary['UNK']
        count["UNK"] += 1
      data.append(index)

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    count = list(count.items())
    return data, count, reversed_dictionary

  def read_data(self,filename):
    with open(filename, encoding="utf-8") as f:
      data = f.read().split()
      data = [x for x in data if x != 'eoood']
    return data

  def build_dataset(self,words, n_words):
    """Process raw inputs into a ."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    self.dictionary = dictionary
    return data, count, reversed_dictionary

  def generate_batch(self, window_size, batch_size, count, verbose=False):
    data = self.train_data
    span = 2 * window_size + 1
    context = np.ndarray(shape=(batch_size,2 * window_size), dtype=np.int64)
    labels = np.ndarray(shape=(batch_size), dtype=np.int64)
    pos_pair = []

    if verbose:
      print("self.data_index: %d, span: %d, len(data): %d" % (self.data_index, span, len(data)))
    if self.data_index + span > len(data):
      self.data_index = 0
      self.process = False
    buffer = data[self.data_index:self.data_index + span]
    pos_u = []
    pos_v = []

    for i in range(batch_size):
      self.data_index += 1
      context[i,:] = buffer[:window_size]+buffer[window_size+1:]
      labels[i] = buffer[window_size]
      if verbose:
        print("self.data_index: %d, span: %d, len(data): %d" % (self.data_index, span, len(data)))
      if self.data_index + span > len(data):
        buffer[:] = data[:span]
        self.data_index = 0
        self.process = False
      else:
        buffer = data[self.data_index:self.data_index + span]

      for j in range(span-1):
        pos_u.append(labels[i])
        pos_v.append(context[i,j])
    neg_v = np.random.choice(self.my_sample_table, size=(batch_size * 2 * window_size, count), p=self.my_ratio)
    return np.array(pos_u), np.array(pos_v), neg_v

