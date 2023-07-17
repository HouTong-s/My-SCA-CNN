import random
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchtext.vocab import build_vocab_from_iterator
from nltk import wordpunct_tokenize
from create_vocab import create_vocab_for_Flickr8k
from torch.nn.utils.rnn import pad_sequence
def default_loader(path):
    return Image.open(path).convert('RGB')
class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, cpi,split,transform = None, loader=default_loader):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'train', 'val', or 'test'
        :param transform: image transform pipeline
        :param cpi: captions per image
        """
        self.split = split
        self.data_folder = data_folder
        assert self.split in {'train', 'val', 'test'}

        #没有词汇表则建立词汇表
        if not os.path.exists("vocab.pt"):
            create_vocab_for_Flickr8k()
        voacb = torch.load("vocab.pt")
            
        imgs = [] # 用于存储图片路径
        captions = [] # 用于存储文本描述
        caption_len = []
        #按照传入的路径和txt文本参数，以只读的方式打开这个文本
        with open(os.path.join(data_folder, self.split + '.txt'), 'r',encoding='utf-8') as fh:
            #跳过第一行
            next(fh)
            for index, line in enumerate(fh): # 使用enumerate()函数获取索引和值
                line = line.strip('\n')
                line = line.rstrip('\n')
                                # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
                words = line.split(',', 1) # 只分割第一个逗号
                            #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
                if index % 4 == 0: # 如果索引能被4整除，说明是新的图片路径
                    imgs.append(words[0]) # 把图片路径添加到imgs列表中
                #分词
                tokens = wordpunct_tokenize(words[1])
                #记录caption的token长度(需要加上首尾特殊符号)
                caption_len.append(torch.LongTensor([len(tokens)+2]))
                #添加首尾特殊字符
                captions.append(torch.LongTensor([voacb["<sos>"]]+[voacb[token] for token in tokens]+[voacb["<eos>"]])) # 把文本描述添加到captions列表中
            
        self.imgs = imgs
        self.captions = pad_sequence(captions, batch_first= True, padding_value=0)
        self.caption_len = caption_len
        self.transform = transform
        self.loader = loader
        self.cpi = cpi
        # Total number of datapoints
        self.dataset_size = len(self.captions)

    #这里的参数index是caption的下标，而不是imgs的下标
    def __getitem__(self, index):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        imgs_index = index//4
        img = self.loader(os.path.join(self.data_folder,"Images",self.imgs[imgs_index]))
        if self.transform is not None:
            img = self.transform(img)
        #  # Randomly select a caption for this image
        # start_idx = index * self.cpi # The start index of the captions for this image
        # end_idx = (index + 1) * self.cpi # The end index of the captions for this image
        # caption_idx = random.randint(start_idx, end_idx - 1) # Randomly select an index from the range
        caption_tensor = torch.LongTensor(self.captions[index])

        caplen = self.caption_len[index]

        if self.split == 'train':
            #print(img.shape, caption_tensor, caplen)
            return img, caption_tensor, caplen
        else:
            start_idx = imgs_index * self.cpi # The start index of the captions for this image
            end_idx = (imgs_index + 1) * self.cpi  # The end index of the captions for this image
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = self.captions[start_idx:end_idx]
            return img, caption_tensor, caplen, all_captions

    def __len__(self):
        return self.dataset_size
