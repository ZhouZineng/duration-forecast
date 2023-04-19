import time

from model import BertTextModel_last_layer, BertTextModel_encode_layer
from utils import MyDataset,read_data
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from config import parsers
import time
import pandas as pd
import os

def load_model(model_path, device):
    model = BertTextModel_last_layer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def text_class_name(texts, pred, args):
    results = torch.argmax(pred, dim=1)
    results = results.cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    if len(results) != 1:
        for i in range(len(results)):
            print(f"文本：{texts[i]}\t预测的类别为：{classification_dict[results[i]]}")
    else:
        print(f"文本：{texts}\t预测的类别为：{classification_dict[results[0]]}")


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    predict_base="./model/20221208-220420"
    predict_path=os.path.join(predict_base,"best_model.pth")
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model = load_model(predict_path, device)
    texts=read_data(args.test_file,False)
    print("模型预测结果：")
    predict=[]
    labels=[]
    x = MyDataset(texts, labels=labels,with_labels=False)
    xDataloader = DataLoader(x, batch_size=32, shuffle=False)
    for batch_index, batch_con in enumerate(xDataloader):
        batch_con = tuple(p.to(device) for p in batch_con)
        out = model(batch_con)
        out=torch.squeeze(out)
        predict+=out.cpu().detach().tolist()
        print(predict)

        # text_class_name(texts, pred, args)
    data_1 = pd.read_csv('./data/testA_processed.csv', encoding='utf-8')
    id_list = data_1['id'].values.tolist()
    datatosave = pd.DataFrame({'id':id_list,'label':predict})
    datatosave.to_csv(os.path.join(predict_base,"submission.csv"), index=False)
    end = time.time()
    print(f"耗时为：{end - start} s")
