import pandas as pd
from sklearn import model_selection 

#数据集划分
def dataset_divide():
    data_path="/home/chengshuang/seg_competetion/zzn/pytorch-nlp/03-bert-TextCNN 文本分类/data_new/train_processed_raw.csv"
    all_data=pd.read_csv(data_path)
    train_data,eval_data=model_selection.train_test_split(all_data, test_size=0.1,
                                                        train_size=0.9, random_state=None,
                                                         shuffle=True)
    train_data.to_csv("./train_processed.csv",index=False)
    eval_data.to_csv("./eval_processed.csv",index=False)
dataset_divide()