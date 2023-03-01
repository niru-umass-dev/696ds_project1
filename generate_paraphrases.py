import json
import os
from typing import List

import torch
from transformers import BartForConditionalGeneration, BartTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

input_sentence = "They were there to enjoy us and they were there to pray for us."


def get_paraphrase(input_sentences: List[str]):
    models = [
        {
            'name': 'eugenesiow/bart-paraphrase',
            'imported_model': BartForConditionalGeneration,
            'tokenizer': BartTokenizer
        },
        {
            'name': 'prithivida/parrot_paraphraser_on_T5',
            'imported_model': AutoModelForSeq2SeqLM,
            'tokenizer': AutoTokenizer
        }
    ]
    for model in models:
        model = model['imported_model'].from_pretrained(model['name'])
        tokenizer = model['tokenizer'].from_pretrained(model['name'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        batch = tokenizer(input_sentence, return_tensors='pt')
        generated_ids = model.generate(batch['input_ids'])
        generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print(generated_sentence)


def combine_data():
    annotations = json.load(open('./anno.json', 'r'))
    restructured_data = []
    for split in annotations:
        print(f"split = {split}")
        for example in annotations[split]:
            restructured_item = dict()
            restructured_item['split'] = split
            restructured_item['entity_a'] = example['entity_a']
            restructured_item['entity_b'] = example['entity_b']
            restructured_item['entity_a_uid'] = example['entity_a_uid']
            restructured_item['entity_b_uid'] = example['entity_b_uid']
            restructured_item['refs_a'] = example['entity_a_summary']
            restructured_item['refs_b'] = example['entity_b_summary']
            restructured_item['refs_comm'] = example['common_summary']

            restructured_data.append(restructured_item)

    generated_cont = json.load(open('./predictions-contrastive.json', 'r'))
    generated_comm = json.load(open('./predictions-common.json', 'r'))

    for split in generated_cont:
        split_indices = [i for i, x in enumerate(restructured_data) if x['split'] == split]
        print(split_indices)
        restructured_split = []
        for idx in range(0, len(generated_cont[split]), 2):
            orig_idx = split_indices[idx // 2]
            print(f"orig_idx = {orig_idx}")
            restructured_item = restructured_data[orig_idx]

            gen_example = generated_cont[split][idx]
            restructured_item['gen_A'] = gen_example['prediction']

            gen_example = generated_cont[split][idx + 1]
            restructured_item['gen_B'] = gen_example['prediction']

            gen_example = generated_comm[split][idx]
            restructured_item['gen_comm'] = gen_example['prediction']

            restructured_split.append(restructured_item)

        print(len(restructured_data))
        print(len(restructured_split))
        print(split_indices[0])
        print(split_indices[-1] + 1)
        restructured_data = restructured_data[:split_indices[0]] + restructured_split + restructured_data[
                                                                                        split_indices[-1] + 1:]

    open('combined_data.json', 'w').write(json.dumps(restructured_data))


if __name__ == '__main__':

