import paddle
import paddle.nn as nn
import paddlenlp

from data import Tokenizer

import numpy as np
from visualdl import LogWriter


class BoWModel(nn.Layer):
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        emb_dim = self.embedder.embedding_dim
        self.encoder = paddlenlp.seq2vec.BoWEncoder(emb_dim)
        self.cos_sim_func = nn.CosineSimilarity(axis=-1)

    def get_cos_sim(self, text_a, text_b):
        text_a_embedding = self.forward(text_a)
        text_b_embedding = self.forward(text_b)
        cos_sim = self.cos_sim_func(text_a_embedding, text_b_embedding)
        return cos_sim

    def forward(self, text):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, embedding_dim)
        summed = self.encoder(embedded_text)

        return summed

model = BoWModel(embedder=token_embedding)


# from data import Tokenizer
tokenizer = Tokenizer()
tokenizer.set_vocab(vocab=token_embedding.vocab)

# 相似句对数据读取
text_pairs = {}
with open("text_pair.txt", "r", encoding="utf8") as f:
    for line in f:
        text_a, text_b = line.strip().split("\t")
        if text_a not in text_pairs:
            text_pairs[text_a] = []
        text_pairs[text_a].append(text_b)

    for text_a, text_b_list in text_pairs.items():
        text_a_ids = paddle.to_tensor([tokenizer.text_to_ids(text_a)])

# 查看相似语句相关度
for text_b in text_b_list:
    text_b_ids = paddle.to_tensor([tokenizer.text_to_ids(text_b)])
    print("text_a: {}".format(text_a))
    print("text_b: {}".format(text_b))
    print("cosine_sim: {}".format(model.get_cos_sim(text_a_ids, text_b_ids).numpy()[0]))
    print()

# 使用VisualDL查看句子向量
# 引入VisualDL的LogWriter记录日志
# import numpy as np
# from visualdl import LogWriter
# 获取句子以及其对应的向量
label_list = []
embedding_list = []

for text_a, text_b_list in text_pairs.items():
    text_a_ids = paddle.to_tensor([tokenizer.text_to_ids(text_a)])
    embedding_list.append(model(text_a_ids).flatten().numpy())
    label_list.append(text_a)

    for text_b in text_b_list:
        text_b_ids = paddle.to_tensor([tokenizer.text_to_ids(text_b)])
        embedding_list.append(model(text_b_ids).flatten().numpy())
        label_list.append(text_b)


with LogWriter(logdir='./sentence_hidi') as writer:
    writer.add_embeddings(tag='test', mat=embedding_list, metadata=label_list)