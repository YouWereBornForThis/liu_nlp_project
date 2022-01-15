import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from lstm import ATAE_LSTM
import warnings
warnings.filterwarnings("ignore")


#1. read the data
import xml.etree.cElementTree as ET
import pandas as pd
path = 'Restaurants_Train.xml'
tree = ET.parse(path)
root = tree.getroot()
data = []

word2idx = {}
idx2word = {}
idx = 1

import re
from string import punctuation
for sentence in root.findall('.//aspectTerms/..'):
    text = sentence.find('text').text
    text_lower = text.lower()
    words = text_lower.split()
    for word in words:
        word = re.sub(r"[{}]+".format(punctuation),"",word)
        if word not in word2idx:
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1
    aspectTerms=sentence.find('aspectTerms')
    for aspectTerm in aspectTerms.findall('aspectTerm'):
        term = aspectTerm.get('term')
        polarity = aspectTerm.get('polarity')
        data.append((text, term, polarity))

df = pd.DataFrame(data,columns=['text', 'term', 'polarity'])
df = df[df['polarity'].isin(['positive', 'negative', 'neutral'])]
df['polarity'] = df['polarity'].map(
    {'positive': 1, 'neutral': 0, 'negative': -1})

seqlen = df['text'].apply(lambda x: len(x.split()))
print("max_seqlen", max(seqlen))   #69


#2. convert to the specific form
import time
class ABSA_Dataset(Dataset):   
    def __init__(self, df, word2idx): 
        self.x_data = df[['text', 'term']].values
        self.y_data = df[['polarity']].values
        self.length = len(self.y_data)
        self.word2idx = word2idx
        all_data = []
        df.index = range(len(df))
        for i in range(len(df['text'])):
            text_indices = self.text_to_sequence(df.loc[i,'text'])
            aspect_indices = self.text_to_sequence(df.loc[i,'term'])
            polarity = int(df.loc[i,'polarity']) + 1
            data = {
                'text_indices': text_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
            }
            all_data.append(data)
        self.data = all_data

    def pad_and_truncate(self, sequence, maxlen, dtype='int64'):
        x = (np.ones(maxlen) * 0).astype(dtype)
        trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        x[:len(trunc)] = trunc
        return x

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            word = re.sub(r"[{}]+".format(punctuation),"",word)
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return self.pad_and_truncate(sequence, 69)

    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec

def build_embedding_matrix(word2idx, embed_dim):
    print('loading word vectors...')
    embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
    fname = 'glove.6B.300d.txt'
    word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
    for word, i in word2idx.items():
        vec = word_vec.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    return embedding_matrix






dataset = ABSA_Dataset(df,word2idx) 
print('dataset[0]',dataset[0])

from torch.utils.data import  random_split
testset_len = int(len(dataset) * 0.2)
trainset, testset = random_split(dataset, (len(dataset)-testset_len, testset_len))
valset = testset

train_data_loader = DataLoader(dataset=trainset, batch_size=16, shuffle=True)
test_data_loader = DataLoader(dataset=testset, batch_size=16, shuffle=False)
val_data_loader = DataLoader(dataset=valset, batch_size=16, shuffle=False)


# 3. load the model
device = torch.device('cpu')
embedding_matrix = build_embedding_matrix(word2idx=word2idx, embed_dim=300)
model = ATAE_LSTM(embedding_matrix).to(device)


criterion = torch.nn.CrossEntropyLoss()
_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(_params,  weight_decay=0.01)

# 4. train
import logging
logger  = logging.getLogger()

max_val_acc = 0
max_val_f1 = 0
max_val_epoch = 0
global_step = 0
path = None

from sklearn import metrics
def evaluate_acc_f1(model, data_loader):
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    model.eval()
    with torch.no_grad():
        for i_batch, t_batch in enumerate(data_loader):
            t_inputs = [t_batch[col].to(device) for col in ['text_indices', 'aspect_indices']]
            t_targets = t_batch['polarity'].to(device)
            t_outputs = model(t_inputs)
            n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
            n_total += len(t_outputs)
            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
    acc = n_correct / n_total
    f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
    return acc, f1
o_time = time.time()
for i_epoch in range(10):
    print('epoch: {}'.format(i_epoch))
    n_correct, n_total, loss_total = 0, 0, 0
    model.train()
    for i_batch, batch in enumerate(train_data_loader):
        global_step += 1
        optimizer.zero_grad()
        inputs = [batch[col].to(device) for col in ['text_indices', 'aspect_indices']]
        outputs = model(inputs)
        targets = batch['polarity'].to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        n_total += len(outputs)
        loss_total += loss.item() * len(outputs)
        if global_step % 20 == 0:
            train_acc = n_correct / n_total
            train_loss = loss_total / n_total
            print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

        
    val_acc, val_f1 = evaluate_acc_f1(model, val_data_loader)
    print('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))

    if val_acc > max_val_acc:
        max_val_acc = val_acc
        max_val_epoch = i_epoch


print('best val_acc: {:.4f}, val_f1: {:.4f}'.format(max_val_acc, max_val_epoch))
print("time: ",time.time()-o_time," s")







