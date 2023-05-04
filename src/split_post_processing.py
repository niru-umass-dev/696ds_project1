import numpy as np
import pandas as pd
import json
import nltk
import string


split = json.load(open('data/combined_data_base_split.json', 'r'))
split_ids = json.load(open('data/temporary_dataset_files/base_split/completed_split_sent_ids.json', 'r'))
indexed_split_ids = json.load(open('data/temporary_dataset_files/base_split/indexed_split_sent_dataset.json', 'r'))

def key_name(str):
    
    if str == 'SA':
        return 'source_reviews_a' 
    if str == 'SB':
        return 'source_reviews_b'
    if str == 'RA':
        return 'refs_a' 
    if str == 'RB':
        return 'refs_b' 
    if str == 'RC':
        return 'refs_comm'
    
def parse_word_tokens(s):
    
    if len(words:=nltk.word_tokenize(s.translate(str.maketrans('','', string.punctuation)))) > 0:
        
        return s 
        
    else:
        return ""
    
    
for key, value in split_ids.items():
        
    if len(value) == 0:
        
        # print(key)
        
        index = int(key[1:4])
        source = key_name(key[4:6])
        source_ind = int(key[6:8])
        
        sentence = indexed_split_ids[index][source][source_ind][key]['sentence']
        
        parse_sent = parse_word_tokens(sentence)
        
        
        if parse_sent != "":
            
            dict_index_split_ids = indexed_split_ids[index][source][source_ind]
        
#             print(split[index][source][source_ind])
        
#             print('\n')
        
            final = ""
        
            for k2, v2 in dict_index_split_ids.items():

                if k2 != key:

                    final += " ".join(v2['simple_sents'].values()) + " "

                else:
                    final = final + parse_sent + " "
            
#             print(final)   
#             print('\n')
            
            indexed_split_ids[index][source][source_ind][key]['simple_sents'] = {key+"_000 ":parse_sent}
            split[index][source][source_ind] = final
            
updated_split = json.dumps(split)
updated_index_split = json.dumps(indexed_split_ids)

with open('data/combined_data_base_split_complete.json', 'w') as f:
    f.write(updated_split)
    
with open('data/temporary_dataset_files/base_split/indexed_split_sent_dataset_complete.json', 'w') as f:
    f.write(updated_index_split)
