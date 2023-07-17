import torch
import pickle
from torchtext.vocab import build_vocab_from_iterator
from nltk import wordpunct_tokenize
from datasets import CaptionDataset
tokenizer = wordpunct_tokenize
#total_sentences = open("Flickr8k/captions.txt","r",encoding="utf-8")
# with open("vocab12.pkl", "rb") as f:
#     v = pickle.load(f)
# a = CaptionDataset("Flickr8k",4,"train")
# v = torch.load("vocab123.pt")
v = torch.load("vocab.pt")
# print(v.get_stoi())
# print(v.get_itos())
print(len(v.vocab))
sentences = ["I am a boy.","He looks good."]
tokens = [tokenizer(sentence) for sentence in sentences]
print(tokens)
v = build_vocab_from_iterator(tokens, specials=['<pad>', '<unk>','<sos>','<eos>'])
v.set_default_index(1)

print(v.get_stoi())
ids = [v[token] for token in tokens[0]]
print(ids)
ids = [v[token] for token in tokens[1]]
print(ids)
new_sentence = "I am a boy 😁 . While she is a girl."
new_token = tokenizer(new_sentence)
print(new_token)
ids = [v[token] for token in new_token]
print(ids)

# torch.save(v, "vocab123.pt")
# with open("vocab12.pkl", "wb") as f:
#     pickle.dump(v, f)
# {'b': 5, 'd': 2, '<pad>': 0, '<unk>': 1, 'c': 4, 'a': 3}

# # 建立词汇表
# vocab = build_vocab_from_iterator([tokens], specials=['<unk>', '<pad>'])
# vocab.set_default_index(vocab['<unk>'])

# # 编码
# ids = [vocab[token] for token in tokens]
# print(ids)

# # 向量化
# tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0) # 添加一个批次维度
# print(tensor)


# # 建立词汇表
# vocab = build_vocab_from_iterator(tokens, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
# vocab.set_default_index(vocab['<unk>'])

# # 保存词汇表
# with open("vocab.pkl", "wb") as f:
#     pickle.dump(vocab, f)

# # 或者
# torch.save(vocab, "vocab.pt")

# # 加载词汇表
# with open("vocab.pkl", "rb") as f:
#     vocab = pickle.load(f)

# # 或者
# vocab = torch.load("vocab.pt")
