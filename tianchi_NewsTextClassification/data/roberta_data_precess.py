import torch.utils.data as Data
import torch


class Dataset(Data.Dataset):
    """定义数据集"""

    def __init__(self, dataset, have_label=True):
        self.dataset = dataset
        self.have_label = have_label

    # 必须实现__len__魔法方法
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, i):
        """定义索引方式"""
        text = self.dataset.iloc[i, :]['text']
        text_list = text.split()
        if len(text_list) > 510:
            # head+tail:empirically select the first 128 and the last 382 tokens.
            text_list = text_list[:128] + text_list[-382:]
            text = ' '.join(text_list)
        if self.have_label:
            label = self.dataset.iloc[i, :]['label']
            return text, label
        else:
            return text,


def get_collate_fn(tokenizer, max_len=512):
    """返回collate_fun函数(通过闭包函数引入形参)"""

    def collate_fn(data):
        model_input_names = tokenizer.model_input_names
        sents = [i[0] for i in data]

        # 批量编码句子
        text_token = tokenizer(text=sents,
                               truncation=True,
                               padding='max_length',
                               max_length=max_len,
                               return_token_type_ids=True,
                               return_attention_mask=True,
                               return_tensors='pt')
        result = {}
        for name in model_input_names:
            result[name] = text_token[name]
        if len(data[0]) == 1:
            return result
        else:
            labels = [i[1] for i in data]
            labels = torch.LongTensor(labels)
            result['labels'] = labels
            return result

    return collate_fn
