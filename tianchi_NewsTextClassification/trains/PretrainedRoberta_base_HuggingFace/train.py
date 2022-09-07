from transformers import PreTrainedTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling, AutoConfig
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# 加载训练好的分词器
tokenizer_fast = PreTrainedTokenizerFast(tokenizer_file='tokenizer.json')

# 模仿HuggingFace transformers/models/bert/tokenizaiton_bert.py
tokenizer_fast.add_special_tokens(special_tokens_dict={'bos_token': "<s>",
                                                       'eos_token': '</s>',
                                                       'unk_token': '<unk>',
                                                       'pad_token': '<pad>',
                                                       'cls_token': '<s>',
                                                       'mask_token': '<mask>',
                                                       'sep_token': '</s>'})

dataset = Dataset.from_text('../../data/csv_to_trainTxt/trainTxt_pretrain_model.txt')


def filter_func(data):
    text = data['text']
    return len(text) > 0 and not text.isspace()  # 过滤空白行


def map_func(data):
    batch_encoding = tokenizer_fast(data['text'], truncation=True, padding="max_length", max_length=512)
    return {'input_ids': batch_encoding['input_ids'],
            'attention_mask': batch_encoding['attention_mask'],
            'token_type_ids': batch_encoding['token_type_ids']}


dataset_filter = dataset.filter(filter_func)
dataset_map = dataset_filter.map(map_func, batched=True, batch_size=1000)  # 每次处理1000条数据

# Data collator used for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_fast, mlm=True, mlm_probability=0.15)

config_roberta_base = AutoConfig.from_pretrained('roberta-base')  # 加载预训练模型roberta config参数
config_roberta_base.update({
    'vocab_size': tokenizer_fast.vocab_size
})
print(config_roberta_base)

model = RobertaForMaskedLM(config_roberta_base)  # 从0开始预训练roberta模型
# model = RobertaForMaskedLM.from_pretrained('output_dir/checkpoint-??????')  # 从checkpoint开始训练(之前模型训练中断)
print('No of parameters: ', model.num_parameters())

training_args = TrainingArguments(
    output_dir='output_dir',
    overwrite_output_dir=True,
    num_train_epochs=30.0,
    # max_steps=50,
    per_device_train_batch_size=16,
    save_strategy='epoch',
    disable_tqdm=True  # 是否使用tqdm显示进度(.py运行时设置disable_tqdm=True)
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_map,
    tokenizer=tokenizer_fast
)

trainer.train()

trainer.save_model("save_model")
