import json
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score
import pandas as pd


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


def read_data(filename):
    """读取JSON数据为DataFrame"""
    data_list = []
    with open(filename) as input_data:
        json_content = json.load(input_data)
        for block in json_content:
            text_a = block['query1']
            text_b = block['query2']
            label = block['label']
            if label not in ['0', '1', '2']:
                label = 0
            data_list.append([text_a, text_b, int(label)])
        return pd.DataFrame(data_list, columns=['query1', 'query2', 'label'])


def get_collate_fn(tokenizer, max_len=512):
    """返回collate_fun函数(通过闭包函数引入形参)"""

    def collate_fn(data):
        # data example:[['天价输液费', '输液价格', 0], ['天价输液费', '输液价格', 0], xxxxxx]
        text, text_pair = [i[0] for i in data], [i[1] for i in data]
        labels = [i[2] for i in data]

        # 批量编码句子
        text_t = tokenizer(text=text,
                           text_pair=text_pair,
                           truncation=True,
                           padding=True,
                           max_length=max_len,
                           return_token_type_ids=True,
                           return_attention_mask=True,
                           return_tensors='pt')

        input_ids = text_t['input_ids']
        attention_mask = text_t['attention_mask']
        token_type_ids = text_t['token_type_ids']
        labels = torch.LongTensor(labels)
        # input_ids.shape=[batch_size, seq_len]
        # labels.shape=[bathc_size]
        return input_ids, attention_mask, token_type_ids, labels

    return collate_fn


# 模型验证
def evaluate(model, dataloader, device=torch.device('cpu')):
    model.eval()

    predict_all = []
    labels_all = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in dataloader:
            # 数据设备切换
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            # out.shape=[batch_size, class_num]
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            predict = torch.argmax(out.cpu(), dim=1).tolist()
            predict_all.extend(predict)
            labels_all.extend(labels.tolist())
        valid_acc = accuracy_score(labels_all, predict_all)  # 精度
    return valid_acc


# 预测
def predict(filename_test, tokenizer, model, device):
    model.eval()

    predict_all = []
    with open(filename_test) as input_data:
        json_content = json.load(input_data)
        for block in json_content:
            with torch.no_grad():
                text = [block['query1']]
                text_pair = [block['query2']]
                text_t = tokenizer(text=text, text_pair=text_pair, return_tensors='pt')

                input_ids = text_t['input_ids'].to(device)
                attention_mask = text_t['attention_mask'].to(device)
                token_type_ids = text_t['token_type_ids'].to(device)
                # out.shape=[1, 3]
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                predict_all.append(out.cpu())
    predict_all = torch.cat(predict_all, dim=0).numpy()
    return predict_all
