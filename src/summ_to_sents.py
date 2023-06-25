import json
import nltk
nltk.download('punkt')

# all_datasets = ['base', 'paraphrase', 'similarity', 'selfparaphrase', 'paraphrase_a', 'paraphrase_b', 'paraphrase_synonyms']

# for dataset_name in all_datasets:
#     input_path = f"data/combined_data_{dataset_name}.json"
#     dataset = json.load(open(input_path, 'r'))
#     new_dataset = []
#     for example in dataset:
#         example['refs_a'] = [nltk.sent_tokenize(example['refs_a'][0])]
#         example['refs_b'] = [nltk.sent_tokenize(example['refs_b'][0])]
#         example['refs_comm'] = [nltk.sent_tokenize(example['refs_comm'][0])]
#         new_dataset.append(example)
#     output_path = f"data/sents_data_{dataset_name}.json"
#     json.dump(new_dataset, open(output_path, 'w'))
    

all_datasets = ['base', 'paraphrase', 'similarity', 'selfparaphrase', 'negation']

for dataset_name in all_datasets:
    input_path = f"data/combined_data_{dataset_name}_split_complete.json"
    dataset = json.load(open(input_path, 'r'))
    new_dataset = []
    for example in dataset:
        example['refs_a'] = [nltk.sent_tokenize(example['refs_a'][0])]
        example['refs_b'] = [nltk.sent_tokenize(example['refs_b'][0])] if dataset_name != 'negation' else [nltk.sent_tokenize(example['refs_a_neg'])]
        example['refs_comm'] = [nltk.sent_tokenize(example['refs_comm'][0])]
        new_dataset.append(example)
    output_path = f"data/sents_data_{dataset_name}_split_complete.json"
    json.dump(new_dataset, open(output_path, 'w'))