#Imports
#from collections import Counter
#import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch import optim
#import time
#import tqdm
import json
import re
import pandas as pd
import os
from simplet5 import SimpleT5

CWD = os.getcwd()
DIR_FILL_IN = 'Final Project\\T2G\\My Work\\'
FULL_WD = os.path.join(CWD, DIR_FILL_IN)


def removeQuotes(lst):
    ret = []
    for s in lst:
        if s != '``' and s != "''":
            ret.append(s)
    return ret


def camelCaseSplit(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token = token.replace(')', '')
        token_split = token.split('_')
        for t in token_split:
            #new_d.append(t.lower())
            new_d.append(t)
    return new_d

def g2tPreprocess(raw):
    df = []
    for item in raw:
        graph = 'g2t:'
        for relation in item['relations']:
            graph += ' <H> ' + ' '.join(removeQuotes(relation[0])) + ' <R> '
            graph += ' '.join(camelCaseSplit(relation[1])) + ' <T> '
            graph += ' '.join(removeQuotes(relation[2]))

        ents = [' '.join(removeQuotes(entity)) for entity in item['entities']]
        text = item['text']
        for i in range(len(ents)):
            text = text.replace('<ENT_'+str(i)+'>', ents[i])
        sample = [graph, text]
        df.append(sample)
    return pd.DataFrame(df, columns=['source_text', 'target_text'])


f_train = open(os.path.join(FULL_WD,'json_datasets/train.json'), 'r')
raw_train = json.load(f_train)
f_train.close()

f_test = open(os.path.join(FULL_WD,'json_datasets/test.json'), 'r')
raw_test = json.load(f_test)
f_test.close()

f_dev = open(os.path.join(FULL_WD,'json_datasets/dev.json'), 'r')
raw_dev = json.load(f_dev)
f_dev.close()

train_df = g2tPreprocess(raw_train)
test_df = g2tPreprocess(raw_test)
dev_df = g2tPreprocess(raw_dev)

train_df = train_df.rename({"source_text": "target_text", "target_text": "source_text"}, axis=1)
test_df = test_df.rename({"source_text": "target_text", "target_text": "source_text"}, axis=1)
dev_df = dev_df.rename({"source_text": "target_text", "target_text": "source_text"}, axis=1)

train_df = train_df[train_df.columns[::-1]]
test_df = test_df[test_df.columns[::-1]]
dev_df = dev_df[dev_df.columns[::-1]]

print(train_df.head(10))
print(test_df.head(10))
print(dev_df.head(10))


# instantiate
model = SimpleT5()

# load (supports t5, mt5, byT5 models)
model.from_pretrained("t5","t5-base")



# train
model.train(train_df=train_df, # pandas dataframe with 2 columns: source_text & target_text
            eval_df=dev_df, # pandas dataframe with 2 columns: source_text & target_text
            source_max_token_len = 512, 
            target_max_token_len = 128,
            batch_size = 8,
            max_epochs = 5,
            use_gpu = True,
            outputdir = "outputs",
            early_stopping_patience_epochs = 0,
            precision = 32
            )

