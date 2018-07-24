import jieba
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

sentences = []
for line in open(r"D:\temp\新宋.txt", encoding="utf-8").readlines():
    if line.strip() != "":
        seg_list = jieba.cut(line.strip())
        sentences.append(list(seg_list))


# model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4, compute_loss=True, iter=5, alpha=0.025)
# print(model.wv.most_similar("笑"))
# print(model.wv.most_similar("石越"))

start_lr = 0.025
end_lr = 0.
n_epoch = 5

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

model = Word2Vec(size=100, window=5, min_count=1, workers=4, compute_loss=True)
model.build_vocab(sentences)
losses = []
learning_rate = start_lr
step_size = (start_lr - end_lr) / n_epoch

for i in range(n_epoch):
    trained_word_count, raw_word_count = model.train(sentences, compute_loss=True,
                                                     start_alpha=learning_rate,
                                                     end_alpha=learning_rate,
                                                     total_examples=model.corpus_count,
                                                     epochs=1)
    loss = model.get_latest_training_loss()
    losses.append(loss)
    print(i, loss, learning_rate)
    learning_rate -= step_size

print(model.wv.most_similar("笑"))
print(model.wv.most_similar("石越"))


model.wv.most_similar(['王安石', '石越'], ['司马光'])
model.wv.most_similar("占城")


# sentences = []
# for line in open(r"D:\temp\in_the_name_of_people.txt", encoding="utf-8").readlines():
#     if line.strip() != "":
#         seg_list = jieba.cut(line.strip())
#         sentences.append(list(seg_list))
#
# model = Word2Vec(sentences, size=100, hs=1, min_count=1, window=3)
#
# model.wv.most_similar('李达康')
