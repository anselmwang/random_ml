import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inputdata import Options

class skipgram(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(skipgram, self).__init__()
    self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)   
    self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True) 
    self.embedding_dim = embedding_dim
    self.init_emb()

  def init_emb(self):
    initrange = 0.5 / self.embedding_dim
    self.u_embeddings.weight.data.uniform_(-initrange, initrange)
    self.v_embeddings.weight.data.uniform_(-0, 0)

  def forward(self, u_pos, v_pos, v_neg, batch_size):
    embed_u = self.u_embeddings(u_pos)
    embed_v = self.v_embeddings(v_pos)

    score = torch.mul(embed_u, embed_v)
    score = torch.sum(score, dim=1)
    log_target = F.logsigmoid(score)

    neg_embed_v = self.v_embeddings(v_neg)
    neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
    log_sampled = F.logsigmoid(-neg_score)

    loss = log_target.sum() + log_sampled.sum()
    return -1 * loss / batch_size

  def input_embeddings(self):
    return self.u_embeddings.weight.data.cpu().numpy()

  def save_embedding(self, file_name, id2word):
    embeds = self.u_embeddings.weight.data
    fo = open(file_name, 'w')
    for idx in range(len(embeds)):
      word = id2word(idx)
      embed = ' '.join(embeds[idx])
      fo.write(word+' '+embed+'\n')
