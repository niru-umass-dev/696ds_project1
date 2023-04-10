import json
import os
from src.trivial_baseline_utils import check_written_dataset


## UTILITY FUNCTIONS TO PREP, MODIFY, SANITY-CHECK DATA
# region Data Utilities

def create_entity_lookup(source_dir: str = 'data/source/tripadvisor_json'):
    entity_lookup = dict()

    for source_file in listdir(source_dir):
        if source_file.startswith("."):
            continue
        file_path = os.path.join(source_dir, source_file)
        try:
            hotel_info = json.load(open(file_path, 'r'))['HotelInfo']
            hotel_id = hotel_info['HotelID'] 
        except:
            print(source_file)
            print(file_path)
            exit(0)
        entity_lookup[hotel_id] = hotel_info


    json.dump(entity_lookup, open("data/entity_lookup.json", 'w'))

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

def get_combined_data(anno_path: str = './data/source/cocosum/anno.json', dev_path: str = './data/source/cocosum/dev.json', test_path: str = './data/source/cocosum/test.json', cont_path: str = './data/source/cocosum/predictions-contrastive.json', comm_path: str = './data/source/cocosum/predictions-common.json', entity_lookup_path: str = './data/entity_lookup.json'):
    annotations = json.load(open(anno_path, 'r'))
    entity_lookup = json.load(open(entity_lookup_path, 'r'))
    source_reviews = {
        'dev':json.load(open(dev_path, 'r')),
        'test':json.load(open(test_path, 'r'))
    }
    dev = json.load(open(dev_path, 'r'))
    test = json.load(open(test_path, 'r'))
    restructured_data = []
    for split in annotations:
        # print(f"split = {split}")
        for idx, example in enumerate(annotations[split]):
            restructured_item = dict()
            restructured_item['split'] = split
            entity_a_info = entity_lookup[example['entity_a']]
            entity_b_info = entity_lookup[example['entity_b']]
            
            #entity A ID and name
            restructured_item['entity_a'] = example['entity_a']
            
            if 'Name' in entity_a_info:
                restructured_item['entity_a_name'] = entity_lookup[example['entity_a']]['Name']
            else:
                url = entity_a_info['HotelURL']
                end = url.rindex("-")
                start = url.rindex("-",0,end)
                restructured_item['entity_a_name'] = " ".join(url[start+1:end].split("_"))
            
            #entity B ID and name
            restructured_item['entity_b'] = example['entity_b']
            
            if 'Name' in entity_b_info:
                restructured_item['entity_b_name'] = entity_lookup[example['entity_b']]['Name']
            else:
                url = entity_b_info['HotelURL']
                end = url.rindex("-")
                start = url.rindex("-",0,end)
                restructured_item['entity_b_name'] = " ".join(url[start+1:end].split("_"))
            
            restructured_item['entity_a_uid'] = example['entity_a_uid']
            restructured_item['entity_b_uid'] = example['entity_b_uid']
            restructured_item['refs_a'] = example['entity_a_summary']
            restructured_item['refs_b'] = example['entity_b_summary']
            restructured_item['refs_comm'] = example['common_summary']
            if split != 'train':
                restructured_item['source_reviews_a'] = source_reviews[split][idx]['entity_a_reviews']
                restructured_item['source_reviews_b'] = source_reviews[split][idx]['entity_b_reviews']

            restructured_data.append(restructured_item)

    generated_cont = json.load(open(cont_path, 'r'))
    generated_comm = json.load(open(comm_path, 'r'))

    for split in generated_cont:
        # print(json.dumps(generated_cont))
        split_indices = [i for i, x in enumerate(restructured_data) if x['split'] == split]
        # print(split_indices)
        restructured_split = []
        for idx in range(0, len(generated_cont[split]), 2):
            orig_idx = split_indices[idx // 2]
            # print(f"orig_idx = {orig_idx}")
            restructured_item = restructured_data[orig_idx]

            gen_example = generated_cont[split][idx]
            restructured_item['gen_a'] = gen_example['prediction']

            gen_example = generated_cont[split][idx + 1]
            restructured_item['gen_b'] = gen_example['prediction']

            gen_example = generated_comm[split][idx]
            restructured_item['gen_comm'] = gen_example['prediction']

            restructured_split.append(restructured_item)

        # print(len(restructured_data))
        # print(len(restructured_split))
        # print(split_indices[0])
        # print(split_indices[-1] + 1)
        restructured_data = restructured_data[:split_indices[0]] + restructured_split + restructured_data[split_indices[-1] + 1:]

    return restructured_data

def check_written_dataset(dataset):
  tally = dict()

  for entity_pair, data in enumerate(dataset):
    split = data['split']
    if split not in tally:
      tally[split] = dict()
    if 'count' not in tally[split]:
      tally[split]['count'] = 0
    tally[split]['count'] = tally[split]['count'] + 1
    for key in data:
      if key == 'split':
        continue
      if key not in tally[split]:
        tally[split][key] = 0
      if type(data[key]) is list:
        tally[split][key] = tally[split][key] + len(data[key])
      else:
        tally[split][key] = tally[split][key] + 1

  print(json.dumps(tally, indent=2))

# endregion Data Utilities

if __name__=='__main__':
    
    ## RUN ALL 3 FUNCTIONS TO FULLY CREATE THE DATASET
    create_entity_lookup()
    json.dump(get_combined_data(), open("data/combined_data.json", 'w'))
    add_source_reviews_to_train_split()
    
    # sanity-check the written dataset
    check_written_dataset(json.load(open("data/combined_data.json", 'r')))
    