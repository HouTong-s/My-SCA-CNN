import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence,pad_packed_sequence
sample1 = torch.rand(5,10)  #第一个序列的长度为5
sample2 = torch.rand(4,10)  #第二个序列的长度为4
sample3 = torch.rand(3,10)  #第三个序列的长度为3

sequence = pad_sequence([sample1,sample2,sample3],batch_first = True,
padding_value=0) 

#打印sequence
print(sequence)

#获取每个序列的长度
lengths = [sample1.size(0), sample2.size(0), sample3.size(0)]

#将sequence和lengths打包成packed_sequence对象
packed = pack_padded_sequence(sequence, lengths, batch_first=True)
print("="*100)
#打印unpacked
print(packed.data.shape)
print(packed.data)

#将packed_sequence对象解析回原来的序列
unpacked, _ = pad_packed_sequence(packed, batch_first=True)

print("="*100)
#打印unpacked
print(unpacked)
