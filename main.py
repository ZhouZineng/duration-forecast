import time
from tqdm import tqdm
from config import parsers
from utils import read_data, MyDataset
from torch.utils.data import DataLoader
from model import BertTextModel_encode_layer, BertTextModel_last_layer
from transformers import BertConfig, BertForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
import os
from FGM import FGM
from PGD import PGD
from torch.nn.parallel import DataParallel




def train(model, device, trainLoader, opt, epoch,args):
    model.train()
    loss_sum, count = 0, 0
    for batch_index, batch_con in enumerate(trainLoader):
        batch_con = tuple(p.to(device) for p in batch_con)
        pred = model(batch_con)

        loss = loss_fn(pred, batch_con[-1])*10
        loss.backward()
        fgm.attack()
        pre_adv = model(batch_con)
        loss_adv = loss_fn(pre_adv, batch_con[-1])*10
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
    #     pgd.backup_grad()
    # # 对抗训练
    #     for t in range(K):
    #         pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
    #         if t != K-1:
    #             model.zero_grad()
    #         else:
    #             pgd.restore_grad()
    #         pre_adv = model(batch_con)
    #         loss_adv = loss_fn(pre_adv, batch_con[-1])
    #         loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    #     pgd.restore() # 恢复embedding参数
        opt.step()
        if args.warmup != 0.0:
            scheduler.step()
        loss_sum += loss
        count += 1
        opt.zero_grad()

        if len(trainLoader) - batch_index <= len(trainLoader) % 1000 and count == len(trainLoader) % 1000:
            msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
            print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
            loss_sum, count = 0.0, 0

        if batch_index % 1000 == 999:
            msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
            print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
            loss_sum, count = 0.0, 0
    torch.save(model.state_dict(), os.path.join(path_save, args.save_model_checkpoint))
    print(f"保存当前模型")


def dev(model, device, devLoader,args):
    global acc_min
    model.eval()
    all_true, all_pred = [], []
    for batch_con in tqdm(devLoader):
        batch_con = tuple(p.to(device) for p in batch_con)
        pred = model(batch_con)
        pred=torch.squeeze(pred)

        pred = torch.argmax(pred, dim=1)

        pred_label = pred.cpu().detach().numpy().tolist()
        # pred_label=[int(pred_labels) for pred_labels in pred_label]
        true_label = batch_con[-1].cpu().numpy().tolist()

        all_true.extend(true_label)
        all_pred.extend(pred_label)


    acc = accuracy_score(all_true, all_pred)
    print(f"dev acc:{acc:.4f}")

    if acc > acc_min:
        acc_min = acc
        torch.save(model.state_dict(), os.path.join(path_save, args.save_model_best))
        print(f"以保存最佳模型")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    start = time.time()
    args = parsers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    print(args)
    path_save = os.path.join(args.base_path, time_now)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        print('文件夹创建完成  ' + path_save)

    with open(os.path.join(path_save, "config.txt"), 'w') as f:
        f.write(str(args))
    f.close()

    train_text, train_label = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)

    trainData = MyDataset(train_text, train_label, with_labels=True)
    trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True)

    devData = MyDataset(dev_text, dev_label, with_labels=True)
    devLoader = DataLoader(devData, batch_size=args.batch_size, shuffle=True)

    # 选择模型
    if args.select_model=="base":
        config = BertConfig.from_pretrained(args.bert_pred, num_labels=args.class_num, hidden_dropout_prob=args.dropout)
        model = BertForSequenceClassification.from_pretrained(args.bert_pred, config=config)
        for param in model.bert.parameters():
            param.requires_grad_(True)
        model.to(device)
    elif args.select_model=="last":
        model = BertTextModel_last_layer().to(device)
    elif args.select_model=="encode_layer":
        model = BertTextModel_encode_layer().to(device)
    else:
        model = BertTextModel_encode_layer().to(device)


    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            model= DataParallel(model)


    """fgm"""
    fgm = FGM(model)
    """pgd"""
    # pgd = PGD(model)
    # K = 3
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.15},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    opt = AdamW(optimizer_grouped_parameters, lr=args.learn_rate)
    loss_fn = CrossEntropyLoss()
    # loss_fn=torch.nn.MSELoss(reduction='mean')

    dataset_len = trainData.__len__()
    if args.warmup != 0.0:
        num_train_optimization_steps = dataset_len / args.batch_size * args.epochs
        scheduler = get_linear_schedule_with_warmup(opt, int(num_train_optimization_steps*args.warmup), num_train_optimization_steps)

    acc_min = float("-inf")
    for epoch in range(args.epochs):
        print(f"******epoch{epoch + 1}******")
        train(model, device, trainLoader, opt, epoch,args)
        print(f"******eval{epoch + 1}******")
        dev(model, device, devLoader,args)

    model.eval()
    torch.save(model.state_dict(), os.path.join(path_save, args.save_model_last))
    end = time.time()
    print(f"运行时间：{(end - start) / 60 % 60:.4f} min")
