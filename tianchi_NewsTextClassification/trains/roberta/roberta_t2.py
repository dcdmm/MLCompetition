# %%

import copy
import torch.nn as nn
import torch
import sys
import os
import time
import torch.utils.data as Data
import torch.optim as optim
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR
import logging
import joblib

# Linux下添加此代码(即添加临时模块搜索路径);当前项目路径为默认模块搜索路径(仅pycharm中)
sys.path.append(os.path.abspath(".." + os.sep + ".." + os.sep + ".."))

from tianchi_NewsTextClassification.data.roberta_data_precess import Dataset, get_collate_fn

# %%

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(lineno)d - %(message)s ',
                    filename='rt2.log')


# %%

def set_seed(seed):
    """PyTorch随机数种子设置大全"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # CPU上设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 当前GPU上设置随机种子
        # A bool that, if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # torch.cuda.manual_seed_all(seed) # 所有GPU上设置随机种子


seed = 2022
set_seed(seed)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(msg="device:" + str(device))

# %%

tokenizer = AutoTokenizer.from_pretrained('../PretrainedRoberta_base_HuggingFace/save_model')
config_update = AutoConfig.from_pretrained('../PretrainedRoberta_base_HuggingFace/save_model')
# 更新或添加新的属性
config_update.update({
    "output_hidden_states": True
})
pretrained = AutoModel.from_pretrained('../PretrainedRoberta_base_HuggingFace/save_model', config=config_update)

# %%

train_set = pd.read_csv("../../datasets/train_set.csv", sep='\t')
test_a = pd.read_csv('../../datasets/test_a.csv', sep='\t')

# %%

dataset_test = Dataset(test_a, have_label=False)
dataLoader_test = Data.DataLoader(dataset=dataset_test, batch_size=16, collate_fn=get_collate_fn(tokenizer))

dataset_train = Dataset(train_set, have_label=True)
dataLoader_train = Data.DataLoader(dataset=dataset_train, shuffle=True, batch_size=16,
                                   collate_fn=get_collate_fn(tokenizer))


# %%

class BertLastFour_MeanMaxPool(torch.nn.Module):
    """Bert最后四层隐藏层的连接 + [MeanPool, MaxPool](transformer实现训练过程)"""

    def __init__(self, pretrained_model, dropout_ratio=0.3):
        super().__init__()
        self.pretrained = pretrained_model
        self.hidden_size = pretrained_model.config.hidden_size
        self.linear1 = torch.nn.Linear(self.hidden_size * 8, 1024)
        self.linear2 = nn.Linear(1024, 14)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        model_output = self.pretrained(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        # all_hidden_states.shape=[pretrained_model.config.num_hidden_layers + 1, batch_size, sequence_length, self.hidden_size]
        all_hidden_states = torch.stack(model_output.hidden_states)
        # concatenate_pooling.shape=[batch_size, sequence_length, self.hidden_size * 4]
        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]),
            -1)  # 最后四层隐藏层的连接

        mean_pooling = torch.mean(concatenate_pooling, dim=1)
        max_pooling = torch.max(concatenate_pooling, dim=1).values
        result = torch.cat((mean_pooling, max_pooling), 1)
        result = self.linear1(result)
        result = self.norm(result)
        result = self.relu(result)
        result = self.dropout(result)
        # result.shape=[batch_size, num_class]
        result = self.linear2(result)
        return result


# %%

# 损失函数
criterion = torch.nn.CrossEntropyLoss()
model = BertLastFour_MeanMaxPool(copy.deepcopy(pretrained))  # 必须进行深拷贝(pretrained(模型子网络结构)会参与梯度更新)
model = model.to(device)  # 模型设备切换


# %%

def get_parameters(model,
                   encoder_layer_init_lr=2e-5,  # bert模型最后一个encoder结构的学习率
                   multiplier=0.95,  # 衰退因子
                   custom_lr=1e-4):  # 自定义的网络层学习率
    parameters = []
    lr = encoder_layer_init_lr

    # bert-larger共有24个encoder结构(分别为encoder.layer.0, encoder.layer.1, ......, encoder.layer.23)
    # bert-base共有12个encoder结构(分别为encoder.layer.0, encoder.layer.1, ......, encoder.layer.11)
    for layer in range(11, -1, -1):
        layer_params = {
            'params': [param for name, param in model.named_parameters() if f'encoder.layer.{layer}.' in name],
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier  # 上个encoder结构的学习率 = 该encoder结构的学习率 * 衰退因子

    # 自定义网络层:下游任务自定义的网络层(具体任务对应修改)
    custom_params = {
        'params': [param for name, param in model.named_parameters() if
                   'linear1' in name or 'linear2' in name or 'norm' in name],
        'lr': custom_lr
    }
    parameters.append(custom_params)
    return parameters  # 这里bert模型的embedding层未加入优化器(即不参与梯度更新)


parameters = get_parameters(model, 2e-5, 0.95, 1e-4)
# 优化器
optimizer = optim.AdamW(parameters)


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


scheduler_lr = get_linear_schedule_with_warmup(optimizer, 0, len(dataLoader_train) * 10)


# %%

# 模型训练
def train(model, dataloader, criterion, optimizer, device):
    model.train()

    for idx, i in enumerate(dataloader):
        # 数据设备切换
        input_ids = i['input_ids'].to(device)
        attention_mask = i['attention_mask'].to(device)
        token_type_ids = i['token_type_ids'].to(device)
        labels = i['labels'].to(device)

        optimizer.zero_grad()
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        loss = criterion(out, labels)  # 损失值
        loss.backward()
        optimizer.step()
        scheduler_lr.step()

        if idx % 1000 == 0 and idx > 0:
            predict = out.argmax(dim=1).cpu().numpy()
            f1 = f1_score(labels.cpu().numpy(), predict, average='micro')  # 评估指标
            logging.info(msg='| step {:5d} | loss {:8.3f} | f1 {:8.3f} |'.format(idx, loss.item(), f1))


# %%

# 模型预测
def predict(model, dataloader, device):
    model.eval()

    predict_list = []
    with torch.no_grad():
        for i in dataloader:
            # 数据设备切换            
            input_ids = i['input_ids'].to(device)
            attention_mask = i['attention_mask'].to(device)
            token_type_ids = i['token_type_ids'].to(device)

            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
            predict_list.append(out.cpu())

    predict_all = torch.cat(predict_list, dim=0)  # 合并所有批次的预测结果
    return predict_all


# %%

EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(model, dataLoader_train, criterion, optimizer, device)
    logging.info(msg='-' * 37)
    logging.info(msg='| end of epoch {:5d} | time: {:5.2f}s |'.format(epoch, time.time() - epoch_start_time))
    logging.info(msg='-' * 37)

# %%


result_pro = predict(model, dataLoader_test, device)
joblib.dump(result_pro.numpy(), 'roberta_result_pro_t2.pkl')
logging.info(msg="预测概率矩阵保存成功")

# %%

pre_result_label = np.argmax(result_pro.numpy(), axis=1)
pre_result_label = pd.DataFrame(pre_result_label, columns=['label'])
pre_result_label.to_csv('../../predict_result/roberta_t2.csv', index=False)
logging.info(msg="模型预测结束")
