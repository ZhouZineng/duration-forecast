from tqdm import tqdm
from config import parsers
# transformer库是一个把各种预训练模型集成在一起的库，导入之后，你就可以选择性的使用自己想用的模型，这里使用的BERT模型。
# 所以导入了bert模型，和bert的分词器，这里是对bert的使用，而不是bert自身的源码。
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch


label_to_class = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
                  10: 7, 11: 7, 12: 8, 13: 9, 14: 9, 15: 9, 16: 9, 17: 9, 18: 10,
                  19: 10, 20: 10, 21: 10, 22: 10, 23: 10, 24: 10, 25: 10,
                  26: 10, 27: 10, 28: 10, 29: 10, 30: 10, 33: 10}
def read_data(file,train=True):
    # 读取文件
    # all_data = open(file, "r", encoding="utf-8").read().split("\n")
    all_data = pd.read_csv(file)
    # 得到所有文本、所有标签、句子的最大长度

    texts, labels = [], []
    desc="Load Test Data"
    if train==True:
        desc="Load Train Data"

    # for idx in range(len(all_data)):
    for idx in tqdm(range(len(all_data)),desc=desc):
        if isinstance(all_data.loc[idx, 'sentence_chn'], str):
            sentence_chn = all_data.loc[idx, 'sentence_chn']
        if isinstance(all_data.loc[idx, 'shenli_op_chn'], str):
            shenli_op_chn = all_data.loc[idx, 'shenli_op_chn']
        if isinstance(all_data.loc[idx, 'court_op_chn'], str):
            court_op_chn = all_data.loc[idx, 'court_op_chn']
        if(train):
            label = all_data.loc[idx, 'label']
            # label =label_to_class[label]
            labels.append(label)
        text = sentence_chn + shenli_op_chn + court_op_chn
        texts.append(text)
    if(train):
        return texts, labels
    return texts


class MyDataset(Dataset):
    def __init__(self, texts, labels, with_labels=True):
        self.all_text = texts
        self.all_label = labels
        # print(labels)
        self.max_len = parsers().max_len
        self.with_labels = with_labels
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)

    def __getitem__(self, index):
        text = self.all_text[index]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(text,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.max_len,
                                      return_tensors='pt')  # Return torch.Tensor objects

        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids  torch.Size([max_len])
        attn_masks = encoded_pair['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values  torch.Size([max_len])
        token_type_ids = encoded_pair['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens  torch.Size([max_len])

        if self.with_labels:  # True if the dataset has labels
            label = int(float(self.all_label[index]))
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

    def __len__(self):
        # 得到文本的长度
        return len(self.all_text)

class Dataset(torch.utils.data.Dataset):
     def __init__(self, path_to_file, train=True):
        self.dataset = pd.read_csv(path_to_file)
        self.train = train

     def __len__(self):
        return len(self.dataset)

     def __getitem__(self, idx):
        sentence_chn = ''
        shenli_op_chn = ''
        court_op_chn = ''
        if isinstance(self.dataset.loc[idx, 'sentence_chn'], str):
            sentence_chn = self.dataset.loc[idx, 'sentence_chn']
        if isinstance(self.dataset.loc[idx, 'shenli_op_chn'], str):
            shenli_op_chn = self.dataset.loc[idx, 'shenli_op_chn']
        if isinstance(self.dataset.loc[idx, 'court_op_chn'], str):
            court_op_chn = self.dataset.loc[idx, 'court_op_chn']
        text = sentence_chn + shenli_op_chn + court_op_chn
        if self.train:
            label = self.dataset.loc[idx, 'label']
            # 根据 idx 分别找到 text 和 label
            sample = {"text": text, "label": label}
            # 返回一个 dict
            return text,label
        else:
            return text



if __name__ == "__main__":
    # train_text, train_label = read_data("./data/train.txt")
    # print(train_text[0], train_label[0])
    # trainDataset = MyDataset(train_text, labels=train_label, with_labels=True)
    # trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)
    # for i, batch in enumerate(trainDataloader):
    #     print(batch[0], batch[1], batch[2], batch[3])
    data_path = "./data/train_processed.csv"
    test_path = "./data/testA_processed.csv"
    train_text, train_label=read_data(data_path)
    print(len(train_text))
    print(len(train_label))
    # dataset = Dataset(data_path,True)
    # print(dataset[0],dataset[1])
    # print(len(dataset), dataset[0])
    # dataset = Dataset(test_path,False)
    # print(len(dataset), dataset[0])
