import numpy as np
import pandas as pd
import json

# # df = pd.read_excel('data/negation_annotation.xlsx', index_col=0)
# df = pd.read_csv('data/negation_annotation.csv', index_col=0)
# original = json.load(open('data/combined_data_base_split_complete.json', 'r'))

# for i in range(0, 48):
#     example = df.loc[(df['example_no'] == i) & (df['summ_type'] == 'refs_a')]
#     neg_list = example['negated_simple_sent'].tolist()

#     series = pd.Series(neg_list)
#     neg_list = series.fillna('').tolist()

#     original[i]['refs_a_neg'] = " ".join(neg_list)

# with open('data/combined_data_negation_split_complete.json', 'w') as f:
#     json.dump(original, f)



## Concatenate split negated sentences to get for non-split NLI results

df = pd.read_csv('data/negation_annotation.csv', index_col=0)

dataset = []
for i in range(0, 48):
    example_df = df.loc[(df['example_no'] == i) & (df['summ_type'] == 'refs_a')]
    
    sent_ids = example_df['sent_id'].tolist()
    num_sents = len(set([int(sent_id[-3:]) for sent_id in sent_ids]))
    sents_list = []
    for sent_no in range(0, num_sents):
        sent_id = f"E{i:03d}RA00N{sent_no:03d}"
        sent_df = example_df.loc[df['sent_id'] == sent_id]
        
        split_sent_ids = sent_df['simple_sent_id'].tolist()
        split_sent_text = sent_df['simple_sent'].tolist()
        split_sent_nos = [int(split_sent_id.split("_")[1]) for split_sent_id in split_sent_ids]
        split_sents = zip(split_sent_nos, split_sent_text)
        split_sents = sorted(split_sents, key=lambda x: x[0])
        
        sent_string = " ".join([text for no, text in split_sents])
        sents_list.append(sent_string)
    
    example_dict = {}
    example_dict['refs_b'] = sents_list
    dataset.append(example_dict)
    
with open('data/combined_data_negation.json', 'w') as f:
    json.dump(dataset, f)