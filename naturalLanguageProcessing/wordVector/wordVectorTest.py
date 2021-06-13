from paddlenlp.embeddings import TokenEmbedding

# 引入VisualDL的LogWriter记录日志
from visualdl import LogWriter

# 初始化TokenEmbedding， 预训练embedding未下载时会自动下载并加载数据
token_embedding = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300")

# # 查看token_embedding详情
# print(token_embedding)

# # 测试
# test_token_embedding = token_embedding.search("中国")
# print(test_token_embedding)

# # 测试
# score1 = token_embedding.cosine_sim("女孩", "女人")
# score2 = token_embedding.cosine_sim("女孩", "书籍")
# print('score1:', score1)
# print('score2:', score2)

####

# 获取词表中前1000个单词
labels = token_embedding.vocab.to_tokens(list(range(0, 1000)))
# 取出这1000个单词对应的Embedding
test_token_embedding = token_embedding.search(labels)

# # 引入VisualDL的LogWriter记录日志
# from visualdl import LogWriter

with LogWriter(logdir='./token_hidi') as writer:
    writer.add_embeddings(tag='test', mat=[i for i in test_token_embedding], metadata=labels)