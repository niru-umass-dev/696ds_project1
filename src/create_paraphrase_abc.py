import json


def get_paraphrase_x_dataset(summ_a_data, summ_b_data, summ_c_data):
    paraphrase_x_dataset = []
    for example_a, example_b, example_c in zip(summ_a_data, summ_b_data, summ_c_data):
        example_new = {}
        example_new['split'] = example_a['split']
        example_new['refs_a'] = [example_a['refs_a'][0]]
        example_new['refs_b'] = [example_b['refs_b'][0]]
        example_new['refs_comm'] = [example_c['refs_comm'][0]]
        paraphrase_x_dataset.append(example_new)
    return paraphrase_x_dataset

# dataset_combos = [
#     ("combined_data_base", "combined_data_paraphrase"),
#     ("combined_data_base_split_complete", "combined_data_paraphrase_split_complete"),
# ]

base = json.load(open(f"data/combined_data_base.json",'r'))
paraphrase = json.load(open(f"data/combined_data_paraphrase.json",'r'))
base_split_complete = json.load(open(f"data/combined_data_base_split_complete.json",'r'))
paraphrase_split_complete = json.load(open(f"data/combined_data_paraphrase_split_complete.json",'r'))

## PARAPHRASE A
paraphrase_a = get_paraphrase_x_dataset(paraphrase, base, base)
json.dump(paraphrase_a, open("data/combined_data_paraphrase_a.json", "w"))

paraphrase_a_split_complete = get_paraphrase_x_dataset(paraphrase_split_complete, base_split_complete, base_split_complete)
json.dump(paraphrase_a_split_complete, open("data/combined_data_paraphrase_a_split_complete.json", "w"))

## PARAPHRASE B
paraphrase_b = get_paraphrase_x_dataset(base, paraphrase, base)
json.dump(paraphrase_b, open("data/combined_data_paraphrase_b.json", "w"))

paraphrase_b_split_complete = get_paraphrase_x_dataset(base_split_complete, paraphrase_split_complete, base_split_complete)
json.dump(paraphrase_b_split_complete, open("data/combined_data_paraphrase_b_split_complete.json", "w"))

## PARAPHRASE C
paraphrase_c = get_paraphrase_x_dataset(base, base, paraphrase)
json.dump(paraphrase_c, open("data/combined_data_paraphrase_c.json", "w"))

paraphrase_c_split_complete = get_paraphrase_x_dataset(base_split_complete, base_split_complete, paraphrase_split_complete)
json.dump(paraphrase_c_split_complete, open("data/combined_data_paraphrase_c_split_complete.json", "w"))
