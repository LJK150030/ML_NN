#%%

import pandas as pd
from simplet5 import SimpleT5

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
import json
import new_preprocessing
import numpy as np

def predict_samples(model_path, test_data_path, num_samples=10):
    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    test_df = new_preprocessing.json_to_df(test_data,
                                           task_prompt='g2t: ',
                                           source_name='graph',
                                           target_name='text')
    model = SimpleT5()
    model.load_model('t5', model_path, use_gpu=True)
    
    for i in range(num_samples):
        source = test_df['source_text'][i]
        prediction = model.predict(source)
        ground_truth = test_df['target_text'][i]
        print(f'source      : {source}')
        print(f'prediction  : {prediction[0]}')
        print(f'ground truth: {ground_truth}')
        print('')

if __name__ == '__main__':
    predict_samples(
        r'outputs\simplet5-epoch-4-train-loss-1.0188-val-loss-1.3412', 
        'json_datasets/out_test.json')
