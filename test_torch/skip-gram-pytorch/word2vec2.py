import torch
from torch.autograd import Variable
import torch.optim as optim
import time
import random

import sys
import logging
from inputdata import Options
from model import skipgram
import logging

class word2vec:
  def __init__(self, inputfile, val_fn, vocabulary_size=4000, embedding_dim=100, epoch_num=2, batch_size=16, windows_size=5,neg_sample_num=10):
    logger = logging.getLogger()
    logger.info("Load train data")
    self.op = Options(inputfile, vocabulary_size)
    logger.info("Load test data")
    self.val_op = Options(val_fn, vocabulary_size, dictionary=self.op.dictionary)
    self.embedding_dim = embedding_dim
    self.windows_size = windows_size
    self.vocabulary_size = vocabulary_size
    self.batch_size = batch_size
    self.epoch_num = epoch_num
    self.neg_sample_num = neg_sample_num

  def train(self):
    model = skipgram(self.vocabulary_size, self.embedding_dim)
    if torch.cuda.is_available():
      model.cuda()

    #return model
    optimizer = optim.SGD(model.parameters(),lr=0.2)
    for epoch in range(self.epoch_num):
      epoch_start = time.time()
      start = time.time()
      self.op.process = True
      batch_num = 0
      batch_new = 0

      while self.op.process:
        pos_u, pos_v, neg_v = self.op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_num, verbose=False)

        pos_u = Variable(torch.LongTensor(pos_u))
        pos_v = Variable(torch.LongTensor(pos_v))
        neg_v = Variable(torch.LongTensor(neg_v))

        if torch.cuda.is_available():
          pos_u = pos_u.cuda()
          pos_v = pos_v.cuda()
          neg_v = neg_v.cuda()

        optimizer.zero_grad()
        loss = model(pos_u, pos_v, neg_v,self.batch_size)
        loss.backward()
        optimizer.step()

        if batch_num%2000 == 0:
          end = time.time()
          with torch.no_grad():
            total_val_loss = 0.
            n_val_batch = 0
            self.val_op.process = True
            while self.val_op.process:
              pos_u, pos_v, neg_v = self.val_op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_num)
              pos_u = Variable(torch.LongTensor(pos_u))
              pos_v = Variable(torch.LongTensor(pos_v))
              neg_v = Variable(torch.LongTensor(neg_v))
              if torch.cuda.is_available():
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()
              val_loss = model(pos_u, pos_v, neg_v, self.batch_size)
              total_val_loss += val_loss.item()
              n_val_batch += 1

          word_embeddings = model.input_embeddings()
          print('epoch,batch=%2d %5d:  pair/sec = %4.2f loss=%4.3f val_loss=%4.3f\r'\
            %(epoch, batch_num, (batch_num-batch_new)*self.batch_size/(end-start),loss.item(), total_val_loss/n_val_batch))
          batch_new = batch_num
          start = time.time()
        batch_num = batch_num + 1
      print("epoch stat, time: %.2f, batch_num: %d" % (time.time()-epoch_start, batch_num))
      torch.save(model.state_dict(), './tmp/skipgram.epoch{}'.format(epoch))
    print("Optimization Finished!")

logging.basicConfig(format='%(asctime)s %(name)s:%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger()

TRAIN_FN = r"d:\temp\xinsong_seg.txt"
VAL_FN = r"d:\temp\xinsong_seg_val.txt"

logger.info("load raw text")

# import jieba
# sentences = []
# for line in open(r"D:\temp\新宋.txt", encoding="utf-8").readlines()[:50000]:
#     if line.strip() != "":
#         seg_list = jieba.cut(line.strip())
#         sentences.append(" ".join(seg_list))
#
# logger.info("prepare train & val")
# random.seed(0)
# random.shuffle(sentences)
# with open(TRAIN_FN, "w", encoding="utf-8") as out_f:
#   out_f.write("\n".join(sentences[:10000]))
#
# with open(VAL_FN, "w", encoding="utf-8") as out_f:
#   out_f.write("\n".join(sentences[10000:12000]))
#
if __name__ == '__main__':
  logger.info("create word2vec")
  wc= word2vec(TRAIN_FN, VAL_FN)
  logger.info("start train")
  model = wc.train()
  logger.info("finish train")
  model.load_state_dict(torch.load('./tmp/skipgram.epoch{}'.format(1)))
  word_embeddings = model.input_embeddings()

  import math
  def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
      x = v1[i];
      y = v2[i]
      sumxx += x * x
      sumyy += y * y
      sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)

  vec0 = word_embeddings[wc.op.dictionary["石越"]]
  sim_list = []
  for i in range(len(wc.op.dictionary)):
    vec1 = word_embeddings[i]
    sim_list.append(cosine_similarity(vec0, vec1))

  reversed_dictionary = dict(zip(wc.op.dictionary.values(), wc.op.dictionary.keys()))

  import numpy as np
  for idx in np.argsort(np.array(sim_list))[::-1][:10]:
    print(reversed_dictionary[idx], sim_list[idx])


