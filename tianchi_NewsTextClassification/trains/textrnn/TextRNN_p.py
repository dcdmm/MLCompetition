# %%

import torch
import numpy as np
import pandas as pd
import torchtext
import torch.nn as nn
import torch.utils.data as Data
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import time
from torch.optim.lr_scheduler import LambdaLR
import copy
import logging
import joblib

# %%

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(lineno)d - %(message)s ',
                    filename='TextRNN.log')


# %%

def set_seed(seed):
    """PyTorch随机数种子设置大全"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # CPU上设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 当前GPU上设置随机种子
        # torch.cuda.manual_seed_all(seed) # 所有GPU上设置随机种\


seed = 2022
set_seed(seed)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(msg="device:" + str(device))

# %%

train_df = pd.read_csv('../../datasets/train_set.csv', sep='\t')
test_df = pd.read_csv('../../datasets/test_a.csv', sep='\t')

# %%

# 加载word2vec字典
load_vocal = joblib.load('../../data/vocab/vocab_word2vec.pkl')

logging.info(msg=str(load_vocal.get_stoi().get('349', 0)))
logging.info(msg=str(load_vocal.get_stoi().get('3113', 0)))
logging.info(msg=str(load_vocal.get_stoi().get('4806', 0)))


# %%

def split_truncate_pad(string,
                       num_steps,  # 句子最大长度
                       stoi,  # Dictionary mapping tokens to indices.
                       padding_index):  # 填充字符'<pad>'在词典中的索引
    """截断或填充文本序列"""
    # 获取字在Vocab对象中的位置
    line = [stoi.get(word, 0) for word in string.split()]
    if len(line) > num_steps:
        # 直接返回列表速度较快
        return line[:4000]
        # return line[:num_steps]  # 句子截断
    return line + [padding_index] * (num_steps - len(line))  # 句子填充


X_train_data = train_df['text'].apply(split_truncate_pad, num_steps=4000,
                                      stoi=load_vocal.get_stoi(), padding_index=1)  # 这里设置句子最大长度为4000
X_test_data = test_df['text'].apply(split_truncate_pad, num_steps=4000, stoi=load_vocal.get_stoi(),
                                    padding_index=1)
y_train = train_df['label'].values

# %%

# 加载预训练词向量文件
vector = torchtext.vocab.Vectors(name="cnew_300.txt",
                                 cache='../word2vec')

pretrained_vector = vector.get_vecs_by_tokens(load_vocal.get_itos())
logging.info(msg="pretrained_vector.shape:" + str(pretrained_vector.shape))


# %%

class TextRNN_MeanMaxPool(nn.Module):
    """
    TextRNN + [MeanPool, MaxPool]模型的pytorch实现(具体任务对应修改)

    Parameters
    ---------
    num_class : int
       类别数
    vocab_size : int
        单词表的单词数目
    embedding_size : int
        输出词向量的维度大小
    hidden_size : int
        隐含变量的维度大小(权重矩阵W_{ih}、W_{hh}中h的大小)
    num_layers : int
        循环神经网络层数
    bidirectional : bool
        是否为设置为双向循环神经网络
    dropout_ratio : float
        元素归零的概率
    """

    def __init__(self, num_class, vocab_size, embedding_size, hidden_size, num_layers, bidirectional, dropout_ratio):
        super(TextRNN_MeanMaxPool, self).__init__()
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          dropout=dropout_ratio,
                          batch_first=True)  # batch_size为第一个维度

        if self.bidirectional:
            mul = 2
        else:
            mul = 1
        self.linear1 = nn.Linear(hidden_size * mul * 2, 1024)
        self.linear2 = nn.Linear(1024, num_class)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()

    def forward(self, text):
        # text.shape=[batch_size, sent len]

        # embedded.shape=[batch_size, sen len, embedding_size]
        embedded = self.dropout(self.embed(text))
        # out.shape=[batch_size, sen len, hidden_size * num directions]  # 即h_{it}
        out, hidden = self.rnn(embedded)
        feat_mean = torch.mean(out, dim=1)
        feat_max = torch.max(out, dim=1).values

        result = torch.cat((feat_mean, feat_max), 1)
        result = self.linear1(result)
        result = self.norm(result)
        result = self.relu(result)
        result = self.dropout(result)
        # result.shape=[batch_size, num_class]
        result = self.linear2(result)
        return result


# %%

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # 学习率预热(线性增加)
            return float(current_step) / float(max(1, num_warmup_steps))
        # 学习率线性衰减(最小为0)
        # num_training_steps后学习率恒为0
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


# %%

class FGM():
    """Fast Gradient Sign Method"""

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self,
               emb_name,  # 添加扰动的embedding层名称
               epsilon=1.0):  # 扰动项中的\epsilon
        for name, param in self.model.named_parameters():
            if param.requires_grad and name == emb_name:
                self.backup[name] = param.detach().clone()
                norm = torch.linalg.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm  # \epsilon * (g / ||g||_2)
                    param.data.add_(r_at)  # embedding层参数增加扰动\Delta x

    def restore(self, emb_name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name == emb_name:
                param.data = self.backup[name]  # 恢复embedding层原有参数值
        self.backup = {}


# %%

# 模型训练
def train(model, dataloader, criterion, optimizer, scheduler, device):
    # FGM对抗训练step 1
    fgm = FGM(model)
    model.train()

    for idx, (text, labels) in enumerate(dataloader):
        # 数据设备切换
        text = text.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(text)
        loss = criterion(out, labels)  # 每个step的损失值
        loss.backward()

        # FGM对抗训练step 2
        fgm.attack(emb_name='embed.weight', epsilon=1.)
        out_adv = model(text)
        loss_adv = criterion(out_adv, labels)
        loss_adv.backward()  # 对抗训练的梯度
        fgm.restore(emb_name='embed.weight')  # 恢复embedding层原有参数值

        optimizer.step()
        scheduler.step()

        if idx % 500 == 0 and idx > 0:  # 每50个step评估一次f1 score
            predict = out.argmax(dim=1).cpu().numpy()
            f1 = f1_score(labels.cpu().numpy(), predict, average='micro')
            logging.info(msg='| step {:5d} | loss {:8.3f} | f1 {:8.3f} |'.format(idx, loss.item(), f1))


# %%

# 模型验证
def evaluate(model, dataloader, device):
    model.eval()

    predict_list = []
    y_true_list = []
    with torch.no_grad():
        for text, labels in dataloader:
            # 数据设备切换
            text = text.to(device)
            labels = labels.to(device)

            out = model(text)
            predict_list.append(out.cpu())
            y_true_list.extend(labels.tolist())

    predict_all = torch.cat(predict_list, dim=0)  # 合并所有批次的预测结果
    y_true_all = torch.tensor(y_true_list)  # 真实标签
    f1 = f1_score(y_true_all.numpy(), predict_all.argmax(dim=1).numpy(), average='micro')  # 验证数据集f1 score
    return f1


# %%

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # 5折分层交叉验证
best_valid_f1_lst = []  # 每折交叉验证最佳模型验证数据集的f1 score
best_model_state_dict_lst = []  # 每折交叉验证最佳模型的状态字典

for fold, (trn_ind, val_ind) in enumerate(skfold.split(X_train_data, y_train)):
    logging.info(msg='*' * 40 + str(fold) + '*' * 40)

    tr_d, va_d = X_train_data.iloc[trn_ind], X_train_data[val_ind]
    tr_y, va_y = y_train[trn_ind], y_train[val_ind]

    dataset_tr = Data.TensorDataset(torch.tensor(tr_d.values.tolist()), torch.tensor(tr_y))
    dataloader_tr = Data.DataLoader(dataset_tr, 64, shuffle=True)
    dataset_va = Data.TensorDataset(torch.tensor(va_d.values.tolist()), torch.tensor(va_y))
    dataloader_va = Data.DataLoader(dataset_va, 64)

    # *************************************************************************************************************
    vocal_size, embedding_size = pretrained_vector.shape
    hidden_size = 256
    dropout = 0.2
    bidirectional = True
    num_class = 14
    num_layers = 2

    net = TextRNN_MeanMaxPool(num_class=num_class,
                              vocab_size=vocal_size,
                              embedding_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout_ratio=dropout,
                              bidirectional=True)
    net.embed.weight.data.copy_(pretrained_vector)  # 使用预训练词向量矩阵(pretrained_vector(tensor张量)不会参与梯度更新))
    net = net.to(device)

    params_1x = [param for name, param in net.named_parameters() if name not in ["embed.weight"]]
    optimizer = torch.optim.Adam([{'params': params_1x, 'lr': 0.001},
                                  {'params': net.embed.parameters(), 'lr': 2e-5}])  # 预训练词向量使用更低的学习率
    scheduler_lr = get_linear_schedule_with_warmup(optimizer, len(dataloader_tr) * 2, len(dataloader_tr) * 5)

    criterion_cross_entropy = nn.CrossEntropyLoss()
    # *************************************************************************************************************

    best_valid_f1 = 0.0  # 最佳模型验证数据集的f1 score
    best_model_state_dict = [None]  # 最佳模型的状态字典
    EPOCHS = 5

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(net, dataloader_tr, criterion_cross_entropy, optimizer, scheduler_lr, device)
        va_f1 = evaluate(net, dataloader_va, device)
        if va_f1 > best_valid_f1:
            best_valid_f1 = va_f1
            best_model_state_dict.pop()
            best_model_state_dict.append(copy.deepcopy(net.state_dict()))  # 必须进行深拷贝
        logging.info(msg='-' * 58)
        logging.info(msg='| end of epoch {:5d} | time: {:5.2f}s | valid f1 {:8.5f} |'.format(epoch,
                                                                                             time.time() - epoch_start_time,
                                                                                             va_f1))
        logging.info(msg='-' * 58)

    best_valid_f1_lst.append(best_valid_f1)
    best_model_state_dict_lst.extend(best_model_state_dict)


# %%

# 模型预测
def predict(model, dataloader, device):
    model.eval()

    predict_list = []
    with torch.no_grad():
        for text, in dataloader:
            # 数据设备切换
            text = text.to(device)
            out = model(text)
            predict_list.append(out.cpu())

    predict_all = torch.cat(predict_list, dim=0)  # 合并所有批次的预测结果
    return predict_all


# %%

dataset_te = Data.TensorDataset(torch.tensor(X_test_data.values.tolist()))
dataloader_te = Data.DataLoader(dataset_te, 64)  # 测试数据集

result_pro = torch.zeros((X_test_data.values.shape[0], 14))
for model_state_dict_i in best_model_state_dict_lst:
    # *************************************************************************************************************
    vocal_size, embedding_size = pretrained_vector.shape
    hidden_size = 256
    dropout = 0.2
    bidirectional = True
    num_class = 14
    num_layers = 2

    net_i = TextRNN_MeanMaxPool(num_class=num_class,
                                vocab_size=vocal_size,
                                embedding_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout_ratio=dropout,
                                bidirectional=True)
    net_i.embed.weight.data.copy_(pretrained_vector)  # 使用预训练词向量矩阵
    net_i = net_i.to(device)

    net_i.load_state_dict(model_state_dict_i)
    # *************************************************************************************************************
    result_pro += predict(net_i, dataloader_te, device) / skfold.n_splits
logging.info(msg="result_pro.shape:" + str(result_pro.shape))

# %%

joblib.dump(result_pro, 'rnn_result_pro.pkl')
logging.info(msg="预测概率矩阵保存成功")

# %%

pre_result_label = np.argmax(result_pro.cpu().numpy(), axis=1)
pre_result_label = pd.DataFrame(pre_result_label, columns=['label'])

# %%

pre_result_label.to_csv('../../predict_result/textrnn.csv', index=False)
logging.info(msg="模型预测结束")
