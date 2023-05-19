import json
from typing import List, Dict, Any
from src.create_dataset import check_written_dataset

def create_similarity_dataset(base_dataset: List[Dict[str, Any]]):
    sim_dataset = []
    for example_no, example in enumerate(base_dataset):
        new_example = {}

        # Since we're comparing an entity with itself, replace all the b's with values of a
        new_example['split'] = example['split']
        new_example['entity_a'] = example['entity_a']
        new_example['entity_b'] = example['entity_a']
        new_example['entity_a_name'] = example['entity_a_name']
        new_example['entity_b_name'] = example['entity_a_name']
        new_example['entity_a_uid'] = example['entity_a_uid']
        new_example['entity_b_uid'] = example['entity_a_uid']
        new_example['refs_a'] = [example['refs_a'][0]]
        new_example['refs_b'] = [example['refs_a'][1]]
        new_example['refs_comm'] = [example['refs_a'][2]]
        new_example['source_reviews_a'] = example['source_reviews_a']
        new_example['source_reviews_b'] = example['source_reviews_a']
        sim_dataset.append(new_example)
    
    return sim_dataset

if __name__ == '__main__':
    base_dataset_path = "data/combined_data_base.json"
    base_dataset = json.load(open(base_dataset_path, 'r'))
    sim_dataset = create_similarity_dataset(base_dataset=base_dataset)
    sim_dataset_path = "data/combined_data_similarity.json"
    
    print()
    print("Checking created dataset")
    print(json.dumps(sim_dataset[0], indent=4))
    
    #writing dataset
    json.dump(sim_dataset, open(sim_dataset_path, 'w'))
    