# coding:utf-8
import re
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def dataprocess(text):
    # 定义正则表达式匹配人名和地名
    pattern = re.compile(r'([\u4e00-\u9fa5]{2,5}?(?:省|自治区|市)){0,1}([\u4e00-\u9fa5]{2,7}?(?:区|县|州)){0,1}([\u4e00-\u9fa5]{2,7}?(?:镇)){0,1}([\u4e00-\u9fa5]{2,7}?(?:村|街|街道)){0,1}([\d]{1,3}?(号)){0,1}')
    # 测试
    # text = '以被告人莫被辉犯运输毒品罪，判处有期徒刑十五年，剥夺政治权利五年，并处没收个人财产20000元，刑期自2011年xx月xx日起至2026年xx月xx日止。,经审理查明，罪犯莫被辉在服刑期间，认罪悔罪，认真遵守法律法规及监规，接受教育改造，积极参加思想、文化、职业技术教育，积极参加劳动，努力完成劳动任务。,本院认为，罪犯莫被辉在服刑改造期间能够认罪服法，确有悔改表现，符合法定减刑条件。但鉴于罪犯莫被辉因贩卖毒品罪，被判处有期徒刑十五年，综合考虑罪犯莫被辉的犯罪性质、社会危害程度、原判刑罚。应再扣减减刑幅度二个月，扣减后对罪犯莫被辉减去有期徒刑一个月，海南省三亚市人民检察院建议对罪犯莫被辉的减刑幅度予以扣减，符合法律规定，应予采纳。'
    pattern2 = re.compile(r'\([^)]*\)')
    text = pattern2.sub('', text)
    text = re.sub(r'[^\u4e00-\u9fa5\d]', '', text)
    text = re.sub(u"\\(.*?\\)", "", text)
    stopwords = ['是', '的', '了', '在', '和','以']
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text =pattern.sub( '', text)
    return text

def static_me(dic):
    for key ,value in zip(dic.keys(),dic.values()):
        max_key=max(max_key,key)
        if value>max_val:
            max_val=value
            max_val_key=key
    print(max_key,max_val_key)

def plt_map(dic):
    keys = list(dic.keys())
    values = list(dic.values())
    plt.plot(keys, values)
    if  os.path.exists('./image1.png'):
        plt.savefig('./image2.png')
        plt.clf()
    else:
        plt.savefig('./image1.png')
        plt.clf()
def read_data(file,train=True):
    # 读取文件
    # all_data = open(file, "r", encoding="utf-8").read().split("\n")
    all_data = pd.read_csv(file)
    # 得到所有文本、所有标签、句子的最大长度

    dic = {}
    desc="Load Test Data"
    if train==True:
        desc="Load Train Data"

    # for idx in range(len(all_data)):
    for idx in tqdm(range(len(all_data)),desc=desc):
        if isinstance(all_data.loc[idx, 'sentence_chn'], str):
            sentence_chn = all_data.loc[idx, 'sentence_chn']
            sentence_chn = dataprocess(sentence_chn)
            all_data.loc[idx, 'sentence_chn']=sentence_chn
        if isinstance(all_data.loc[idx, 'shenli_op_chn'], str):
            shenli_op_chn = all_data.loc[idx, 'shenli_op_chn']
            shenli_op_chn = dataprocess(shenli_op_chn)
            all_data.loc[idx, 'shenli_op_chn']=shenli_op_chn
        if isinstance(all_data.loc[idx, 'court_op_chn'], str):
            court_op_chn = all_data.loc[idx, 'court_op_chn']
            court_op_chn = dataprocess(court_op_chn)
            all_data.loc[idx, 'court_op_chn']=court_op_chn
        text=sentence_chn+shenli_op_chn+court_op_chn
        dic[len(text)] = dic.get(len(text), 0) + 1
        # break
    return all_data,dic
def read_raw_data(file):
    all_data = pd.read_csv(file)
    dic = {}
    for idx in tqdm(range(len(all_data))):
        text=all_data.loc[idx, 'fact']
        dic[len(text)] = dic.get(len(text), 0) + 1
        # break
    return dic


if __name__ == "__main__":
    train_file="./test.csv"
    test_file="./train.csv"
    train_data,dic=read_data(train_file,train=True)
    # plt_map(dic)
    dic=read_raw_data(test_file)
    plt_map(dic)
    train_data.to_csv("./train_processed.csv",index=False)
    eval_data,dic=read_data(test_file,train=False)
    eval_data.to_csv("./test_processed.csv",index=False)
    # plt_map(dic)
