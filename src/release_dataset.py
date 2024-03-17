import json

output_dataset = []

with open("/Users/philipgeorge/projects/caspr/data/sents_data_selfparaphrase.json") as fr:
    syn_low_contrast = json.load(fr)
with open("/Users/philipgeorge/projects/caspr/data/sents_data_negation.json") as fr:
    syn_high_contrast = json.load(fr)
with open("/Users/philipgeorge/projects/caspr/data/sents_data_paraphrase.json") as fr:
    paraphrase = json.load(fr)



new_dataset = []
for idx,_ in enumerate(syn_low_contrast):
    new_entity_pair = {}

    new_entity_pair['entity_a'] = syn_low_contrast[idx]['entity_a']
    new_entity_pair['entity_b'] = syn_low_contrast[idx]['entity_b']
    new_entity_pair['syn_low_contrast'] = " ".join(syn_low_contrast[idx]['refs_b'][0]).strip('\n')
    new_entity_pair['syn_high_contrast'] = " ".join(syn_high_contrast[idx]['refs_b'][0]).strip('\n')
    new_entity_pair['paraphrase_a'] = " ".join(paraphrase[idx]['refs_a'][0]).strip('\n')
    new_entity_pair['paraphrase_b'] = " ".join(paraphrase[idx]['refs_b'][0]).strip('\n')
    new_dataset.append(new_entity_pair)
    del new_entity_pair

with open("/Users/philipgeorge/projects/caspr/data/caspr_synthetic_dataset.json", 'w') as fw:
    json.dump(new_dataset, fw)