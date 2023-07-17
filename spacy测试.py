import spacy
nlp = spacy.load("en_core_web_sm") 

# 使用模型，传入句子即可
doc = nlp("<sos> Apple is looking at buying U.K. startup for $1 billion. <eos>")

# 获取分词结果
tokens = [token.text for token in doc]
print(tokens)