import json
import os
from src.trivial_baseline_utils import check_written_dataset

def add_source_reviews_to_train_split():
    combined_data = json.load(open("data/combined_data.json", 'r'))
    ta_source_folder = "data/source/tripadvisor_json_processed"
    
    new_data = []
    
    for idx,example in enumerate(combined_data):
        if example['split']!='train':
            new_data.append(example)
            continue
        
        # FOR ENTITY A
        entity_a = example['entity_a']
        entity_a_uids = example['entity_a_uid']
        
        source_file = json.load(open(os.path.join(ta_source_folder,f"{entity_a}.json"),'r'))
        all_reviews = source_file['Reviews']
        
        source_reviews_a = []
        found_review_ids = []
        for entity_a_uid in entity_a_uids:
            for review in all_reviews:
                if review['ReviewID']==entity_a_uid:
                    source_reviews_a.append(review['Content'])
                    found_review_ids.append(entity_a_uid)
                    break
        
        if len(source_reviews_a) != len(entity_a_uids):
            print(f"Source Review Missing | example {idx} | Entity A ")
            print(entity_a_uids)
            print(found_review_ids)
            exit(0)
                  
        
        # FOR ENTITY B
        
        entity_b = example['entity_b']
        entity_b_uids = example['entity_b_uid']
        
        source_file = json.load(open(os.path.join(ta_source_folder,f"{entity_b}.json"),'r'))
        all_reviews = source_file['Reviews']
        
        source_reviews_b = []
        found_review_ids = []
        for entity_b_uid in entity_b_uids:
            for review in all_reviews:
                if review['ReviewID']==entity_b_uid:
                    source_reviews_b.append(review['Content'])
                    found_review_ids.append(entity_b_uid)
                    break
        
        if len(source_reviews_b) != len(entity_b_uids):
            print(f"Source Review Missing | example {idx} | Entity B ")
            print(entity_b_uids)
            print(found_review_ids)
            exit(0)
        
        example['source_reviews_a'] = source_reviews_a
        example['source_reviews_b'] = source_reviews_b
        
        new_data.append(example)
    
    json.dump(new_data, open("data/combined_data.json","w"))
    check_written_dataset(new_data)
        

if __name__=='__main__':
    add_source_reviews_to_train_split()
    