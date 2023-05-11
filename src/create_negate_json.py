import numpy as np
import pandas as pd
import json

df = pd.read_excel('data/negation_annotation.xlsx', index_col=0)
original = json.load(open('data/combined_data_base_split_complete.json', 'r'))

for i in range(20,30):

    example = df.loc[(df['example_no'] == i) & (df['summ_type'] == 'refs_a')]
    neg_list = example['negated_simple_sent'].tolist()

    series = pd.Series(neg_list)
    neg_list = series.fillna('').tolist()

    original[i]['refs_a_neg'] = " ".join(neg_list)
    
    
with open('data/combined_data_negation_split_complete.json', 'w') as f:
    json.dump(original[20:30], f)