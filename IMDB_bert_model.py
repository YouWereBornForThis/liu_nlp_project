import warnings
warnings.filterwarnings("ignore")

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
import numpy as np

device = torch.device('cpu')

import pandas as pd

#1. read the data

df= pd.read_csv("IMDB_Dataset.csv")
#print(len(df))
#(df.head)
df = df.drop_duplicates(subset = ['review'], keep = 'first')
#print(len(df))

df.index = range(len(df))
import re
for i in range(len(df)):
    df.loc[i,"review"] = re.sub(r'<br />','',df.loc[i,"review"])

class_map = {
    "positive": 1,
    "negative": 0
}
df['Sentiment'] = df['sentiment'].map(class_map)


train_data = df.sample(frac=0.8)[0:4000]
test_data = df[~df.index.isin(train_data.index)][0:1000]
train_data.index = range(len(train_data))
test_data.index = range(len(test_data))
print(train_data.head,test_data.head)


seqlen = df['review'].apply(lambda x: len(x.split()))
print("max_seqlen", max(seqlen))   #2470

# BERT can only process sentences of up to 512 words due to the limitation of position-embedding
print("90%_text_len", np.percentile(seqlen, 90))   #444
print("75%_text_len", np.percentile(seqlen, 75))   #276
print("50%_text_len", np.percentile(seqlen, 50))   #171


# 2. convert to the bert form
# 'CLS' + tokens_a + 'SEP' + tokens_b + 'SEP'  

# The convention in BERT is: 
# (a) For sequence pairs: #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]  
# type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
# (b) For single sequences: 
# tokens:   [CLS] the dog is hairy . [SEP]#  type_ids: 0   0   0   0  0     0 0 
# (c) for sequence triples: #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
#  type_ids: 0 0 0 0 1 1 0 0 0 0

# 
def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_feature(data, tokenizer,max_seq_length=64,evaluate=False):
    cls_token = '[CLS]'
    sep_token = '[SEP]'
    cls_token_segment_id = 0
    pad_token_segment_id = 0
    pad_token = 0
    features = []

    for i, sequence in enumerate(data['review']):
        length = 0
        length += 2  #[CLS] [SEP]
        flag = 0
        whole_token = [cls_token]
        segment_ids = [cls_token_segment_id] 
        #             [sequence_a_segment_id] * (len(tokens_a)+1) + \
        #             [sequence_b_segment_id] * (len(tokens_b)+1) + \
        #             ([pad_token_segment_id]*padding_length)
        sentence = sequence.split(".") 
        
        for sent in sentence:
            token = tokenizer.tokenize(sent)
            len_token = len(token)
            if length + len_token < max_seq_length:
                whole_token += token +[sep_token]
                length += len_token
                segment_ids += [flag] * (len_token + 1)
                if flag == 0:
                    flag = 1
                else:
                    flag = 0
                length += 1
            else:
                break
        
        input_ids = tokenizer.convert_tokens_to_ids(whole_token)
        
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        
        input_ids += ([pad_token] * padding_length)
        input_mask += [0] * padding_length
        segment_ids += ([pad_token_segment_id]*padding_length)
        # print(sentence)
        # print(whole_token)
        # print(len(input_ids),len(input_mask),len(segment_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append({
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'label_id': data.loc[i,"Sentiment"] if not evaluate else 0
        })
        
    return features


def load_examples(data,  tokenizer, max_seq_length=64, evaluate=False):
    features = convert_examples_to_feature(data, tokenizer, max_seq_length, evaluate)
    
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f['label_id'] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    return dataset

# 3. load the model
print("begin_load_the_model")

from transformers import BertTokenizer,BertConfig,BertForSequenceClassification
configuration = BertConfig() 
model = BertForSequenceClassification(configuration) 
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model.to(device)

# 4. train
print("begin_train")

train_dataset = load_examples(train_data,  tokenizer, max_seq_length=64, evaluate=False)
train_sampler = SequentialSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)
test_dataset = load_examples(test_data,  tokenizer, max_seq_length=64, evaluate=False)
test_sampler = SequentialSampler(train_dataset)
test_dataloader = DataLoader(test_dataset, sampler=train_sampler, batch_size=1)
max_steps = 10000
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

global_step = 0
model.zero_grad()



for i in range(5):
    print('epoch: {}'.format(i))
    n_correct, n_total, loss_total = 0, 0, 0
    for step, batch in enumerate(train_dataloader):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        outputs = model(
            input_ids = batch[0],
            attention_mask = batch[1],
            token_type_ids = batch[2],
            labels = batch[3]
        )
        _, logits = outputs[: 2]
        
        targets = batch[3]
        loss = outputs[0]
        
        loss.backward()
        optimizer.step()
        model.zero_grad()
        global_step += 1
        n_correct += (torch.argmax(logits,-1) == targets).sum().item()
        n_total += 32
        loss_total += loss.item() * 32
        if global_step % 10 == 0:
            train_acc = n_correct / n_total
            train_loss = loss_total / n_total
            print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

from sklearn import metrics
n_correct, n_total, loss_total = 0, 0, 0
t_targets_all, t_outputs_all = None, None
count = 0
for step, batch in enumerate(test_dataloader):
    count += 1
    model.eval()
    with torch.no_grad():
        batch = tuple(t.to(device) for t in batch)
        outputs = model(
            input_ids = batch[0],
            attention_mask = batch[1],
            token_type_ids = batch[2]
        )
        logits = outputs.logits
        targets = batch[3]
        n_correct += (torch.argmax(logits,-1) == targets).sum().item()
        n_total += 1
        if t_targets_all is None:
                t_targets_all = targets
                t_outputs_all = logits
        else:
                t_targets_all = torch.cat((t_targets_all, targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, logits), dim=0)

    if count == 1000:
        break
    # I keep meeting an index error here that I can only solve in this way.
acc = n_correct / n_total
f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1], average='macro')
print("test_acc: ",acc," f1: ",f1)