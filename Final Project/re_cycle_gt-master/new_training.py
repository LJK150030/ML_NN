# %%
import new_preprocessing
from simplet5 import SimpleT5
import pandas as pd
import json


def train(mode='duo'):
    train_data = []
    test_data = []
    with open('json_datasets/out_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('json_datasets/out_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    train_g2t_df = new_preprocessing.json_to_df(train_data,
                                            task_prompt='g2t: ',
                                            source_name='graph',
                                            target_name='text')

    test_g2t_df = new_preprocessing.json_to_df(test_data,
                                           task_prompt='g2t: ',
                                           source_name='graph',
                                           target_name='text')
    
    train_t2g_df = new_preprocessing.json_to_df(train_data,
                                                task_prompt='t2g: ',
                                                source_name='text',
                                                target_name='graph')
    
    test_t2g_df = new_preprocessing.json_to_df(test_data,
                                                task_prompt='t2g: ',
                                                source_name='text',
                                                target_name='graph')

    train_duo_df = pd.concat([train_g2t_df, train_t2g_df], ignore_index=True)
    
    test_duo_df = pd.concat([test_g2t_df, test_t2g_df], ignore_index=True)
    
    model = SimpleT5()
    model.from_pretrained("t5", "t5-small")
    
    if mode == 'duo':
        train_df = train_duo_df
        test_df = test_duo_df
    elif mode == 't2g':
        train_df = train_t2g_df
        test_df = test_t2g_df
    elif mode == 'g2t':
        train_df = train_g2t_df
        test_df = test_g2t_df
    else:
        raise ValueError(f'unknown mode: {mode}')

    model.train(train_df=train_df,  # pandas dataframe with 2 columns: source_text & target_text
                eval_df=test_df,  # pandas dataframe with 2 columns: source_text & target_text
                source_max_token_len=64,
                target_max_token_len=64,
                batch_size=40,
                max_epochs=10,
                use_gpu=True,
                outputdir=f'outputs_{mode}',
                early_stopping_patience_epochs=0,
                precision=32
                )
    

if __name__ == '__main__':
    train(mode='duo')
    train(mode='g2t')
    train(mode='t2g')



