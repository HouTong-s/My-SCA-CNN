import os
import torch
from torchtext.vocab import build_vocab_from_iterator
from nltk import wordpunct_tokenize
def create_vocab_for_Flickr8k():
    fh = open(os.path.join("Flickr8k",'captions.txt'), 'r',encoding='utf-8')
                    #按照传入的路径和txt文本参数，以只读的方式打开这个文本
    captions = [] # 用于存储文本描述
    for index, line in enumerate(fh): # 使用enumerate()函数获取索引和值
        line = line.strip('\n')
        line = line.rstrip('\n')
                        # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
        words = line.split(',', 1) # 只分割第一个逗号
                    #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
        captions.append(words[1]) # 把文本描述添加到captions列表中
    tokens = [wordpunct_tokenize(sentence) for sentence in captions]
    v = build_vocab_from_iterator(tokens, specials=['<pad>','<unk>', '<sos>','<eos>'])
    v.set_default_index(1)
    print(v.get_itos())
    #print(v.get_stoi())
    torch.save(v,"vocab.pt")