#%%

import pandas as pd
from simplet5 import SimpleT5

from collections import defaultdict

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
import json
import new_preprocessing
import numpy as np

#from nubia import Nubia


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




def predict_samples(model_path, test_data_path, num_samples=1000, mode='g2t'):
    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    if mode == 'g2t':
        test_df = new_preprocessing.json_to_df(test_data,
                                            task_prompt='g2t: ',
                                            source_name='graph',
                                            target_name='text')
    else:
        test_df = new_preprocessing.json_to_df(test_data,
                                               task_prompt='t2g: ',
                                               source_name='text',
                                               target_name='graph')
    model = SimpleT5()
    #model.from_pretrained("t5","t5-small")
    model.load_model('t5', model_path, use_gpu=True)

    bleu = Bleu(4)
    meteor = Meteor()
    rouge = Rouge()
    cider = Cider()

    hyp = defaultdict(list)
    
    ref = defaultdict(list)
    ptr = 0
    same = defaultdict(list)
    all_data_len = len(test_df)-1
    for i in range(num_samples):
        source = test_df['source_text'][i]
        prediction = model.predict(source)
        ground_truth = test_df['target_text'][i]
        #print(f'source      : {source}')
        #print(f'prediction  : {prediction[0]}')
        #print(f'ground truth: {ground_truth}')
        #print(f"")

        if i > 0 and test_df['source_text'][i] != test_df['source_text'][i-1]:
            ptr += 1
        same[ptr].append(ground_truth.lower())
        ref[i] = same[ptr]
        hyp[i] = [prediction[0]]

    #print(f"--------- Original Paper Evaluations Metrics --------------")
    ret = bleu.compute_score(ref, hyp)
    print('BLEU INP {0:}'.format(len(hyp)))
    print('BLEU 1-4 {0:}'.format(ret[0]))
    
    print('METEOR {0:}'.format(meteor.compute_score(ref, hyp)[0]))
    print('ROUGE_L {0:}'.format(rouge.compute_score(ref, hyp)[0]))
    print('Cider {0:}'.format(cider.compute_score(ref, hyp)[0]))

    


if __name__ == '__main__':
    
    print('-----Single-Task-t2g------')
    predict_samples(
        r'outputs_t2g\simplet5-epoch-7-train-loss-0.0921-val-loss-0.3717',
        'json_datasets/out_test.json',
        mode='t2g')
    
    print('-----Multi-Task-t2g------')
    predict_samples(
        r'outputs_duo\simplet5-epoch-8-train-loss-0.3623-val-loss-0.8387',
        'json_datasets/out_test.json',
        mode='t2g')
    
    print('-----Single-Task-g2t------')
    predict_samples(
        r'outputs_g2t\simplet5-epoch-9-train-loss-0.795-val-loss-1.3175',
        'json_datasets/out_test.json',
        mode='g2t')

    print('-----Multi-Task-g2t------')
    predict_samples(
        r'outputs_duo\simplet5-epoch-8-train-loss-0.3623-val-loss-0.8387',
        'json_datasets/out_test.json',
        mode='g2t')
