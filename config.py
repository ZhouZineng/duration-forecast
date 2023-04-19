import argparse
import os.path


def parsers():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("./data_new", "train_processed.csv"))
    parser.add_argument("--dev_file", type=str, default=os.path.join("./data_new", "eval_processed.csv"))
    parser.add_argument("--test_file", type=str, default=os.path.join("./data_new", "testA_processed.csv"))
    parser.add_argument("--bert_pred", type=str, default="./roberta-base-finetuned-chinanews-chinese", help="bert 预训练模型")
    parser.add_argument("--select_model", type=str, default="last", help="选择模型,base,last,encode_layer")
    parser.add_argument("--class_num", type=int, default=34, help="分类数")
    parser.add_argument("--max_len", type=int, default=300, help="句子的最大长度")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learn_rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1, help="失活率")
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help="TextCnn 的卷积核大小")
    parser.add_argument("--num_filters", type=int, default=384, help="TextCnn 的卷积输出")
    parser.add_argument("--encode_layer", type=int, default=12, help="chinese bert 层数")
    parser.add_argument("--hidden_size", type=int, default=768, help="bert 层输出维度")
    parser.add_argument("--base_path", type=str, default=os.path.join("model"))
    parser.add_argument("--save_model_checkpoint", type=str, default= "checkpoint_model.pth")
    parser.add_argument("--save_model_best", type=str, default= "best_model.pth")
    parser.add_argument("--save_model_last", type=str, default="last_model.pth")
    parser.add_argument('--warmup', type=float, default=0.15, help='warm up ratio')
    args = parser.parse_args()
    return args
