#%%
import json
import pandas as pd
import re

def camelCaseSplit(identifier):
	matches = re.finditer(
		'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
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

def remove_text_artifacts(text):
    text = text.replace('``', '')
    text = text.replace("''", '')
    text = text.replace('  ', ' ')
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(" '", "'")
    return text

def preprocess(in_filepath, out_filepath):
    all_data = None
    out_data = []
    with open(in_filepath, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    
    for sample in all_data:
        entities = [' '.join(entity) for entity in sample['entities']]
        text = sample['text']
        for i in range(len(entities)):
            text = text.replace(f'<ENT_{i}>', entities[i])
        graph = ''
        for triplet in sample['relations']:
            entity0 = ' '.join(triplet[0])
            entity1 = ' '.join(triplet[2])
            relation = ' '.join(camelCaseSplit(triplet[1]))
            graph_triplet = f'<H> {entity0} <R> {relation} <T> {entity1} '
            graph += graph_triplet
        graph = remove_text_artifacts(graph)
        text = remove_text_artifacts(text)

        
        out_data.append({'text': text, 'graph': graph})
        
        
    with open(out_filepath, 'w', encoding='utf-8') as f:
       json.dump(out_data, f, indent=4)
       

def json_to_df(all_data, task_prompt, source_name, target_name):
    df = pd.DataFrame(all_data)
    df[source_name] = df[source_name].map(lambda x: task_prompt + x)
    
    df = df.rename(columns={source_name: 'source_text', target_name: 'target_text'})
    return df
    
    #for sample in all_data:
    
def preprocess_all():
    dev_filepath = 'json_datasets/dev.json'
    out_dev_filepath = 'json_datasets/out_dev.json'
    preprocess(dev_filepath, out_dev_filepath)
    
    train_filepath = 'json_datasets/train.json'
    out_train_filepath = 'json_datasets/out_train.json'
    preprocess(train_filepath, out_train_filepath)
    
    test_filepath = 'json_datasets/test.json'
    out_test_filepath = 'json_datasets/out_test.json'
    preprocess(test_filepath, out_test_filepath)

if __name__ == '__main__':
    #preprocess_all()
    dev_data = []
    with open('json_datasets/out_dev.json', 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    json_to_df(dev_data, 'g2t: ', 'graph', 'text')

# %%
