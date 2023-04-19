# 项目名称：
使用 Bert + TextCNN 融合模型来对中文进行分类，即文本分类
Bert往往可以对一些表述隐晦的句子进行更好的分类，TextCNN往往对关键词更加敏感。

# 项目环境：
pytorch、python   
相关库安装
`pip install -r requirement.txt`

# 项目目录：
```
bert-TextCNN  
├── bert-base-chinese           bert 中文预训练模型     
├── config.py                   配置文件 
├── data_new                    数据集
│   ├── dataprocess.py          数据预处理
│   ├── dataset_divide.py       数据集划分
│   ├── eval_processed.csv      验证集
│   ├── test_processed.csv      测试集
│   ├── train_processed.csv     训练集
├── FGM.py                      FGM扰动
├── main.py                     main函数
├── model                       模型存储位置
│   └── README.md
├── model.py                    模型文件
├── openclap_localpath          openclap 预训练模型
├── PGD.py                      PGD扰动
├── predict.py                  预测文件
├── requirement.txt             需要的安装包
├── roberta-base-finetuned-chinanews-chinese
└── utils.py                    数据处理文件
```

# bert-TextCNN 模型结构图
## 模型1
Bert-Base除去第一层输入层，有12个encoder层，每个encode层的第一个token（CLS）向量都可以当作句子向量，
我们可以抽象的理解为，encode层越浅，句子向量越能代表低级别语义信息，越深，代表更高级别语义信息。
我们的目的是既想得到有关词的特征，又想得到语义特征，模型具体做法是将第1层到第12层的CLS向量，作为TextCNN的输入，进行文本分类。

## 模型2
将 bert 模型的最后一层的输出的内容作为 TextCNN 模型的输入，送入模型在继续进行学习，得到最终的结果，进行文本分类


# 项目数据集
数据集使用比赛中经过预处理后的train_processed.csv、test_processed.csv、eval_processed.csv，为34分类问题。
其中训练集一共有 45000 条，验证集一共有 2000 条，测试集一共有 25000 条。


# 模型训练
`python main.py`

# 模型预测
`python predict.py`

## 修改内容：
在配置文件中修改长度、类别数、预训练模型地址    
```
parser.add_argument("--select_model_last", type=bool, default=True, help="选择模型")
parser.add_argument("--bert_pred", type=str, default="./bert-base-chinese", help="bert 预训练模型")
parser.add_argument("--class_num", type=int, default=10)   
parser.add_argument("--max_len", type=int, default=38)
```

