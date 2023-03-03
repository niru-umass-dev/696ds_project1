import json
import os
from tqdm.auto import tqdm
import openai
import time
import sys
openai.organization = "org-uqW82WXjsx1QYRwThjvVeQqQ"
openai.api_key = "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"

# openai.api_key = os.getenv("OPENAI_API_KEY")


# print(openai.Model.list())

# input_sentence = "The hotel has an enjoyable ambience and is great value for money. This location is great if you're looking to access the great fun and action close-by, but with that comes hustle-and-bustle noise from the street the hotel is located on. The rustic rooms here are a little small but really comfortable and clean while the beds are big. There was a thatched roof to the room that provided an airy feel but of course not soundproof. The hotel provides great breakfast to be eaten above the garden or to take away. The hotel does not provide private parking unfortunately."

def get_paraphrase(input_sentence: str):
    prompt = f"Paraphrase this: f'{input_sentence}'"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=512,
    )
    return response['choices'][0]['text']


def combine_data():
    annotations = json.load(open('./anno.json', 'r'))
    restructured_data = []
    for split in annotations:
        # print(f"split = {split}")
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
        restructured_data = restructured_data[:split_indices[0]] + restructured_split + restructured_data[
                                                                                        split_indices[-1] + 1:]

        return restructured_data

    open('combined_data.json', 'w').write(json.dumps(restructured_data))


if __name__ == '__main__':

    # combined_data = combine_data()

    combined_data = json.load(open('./combined_data.json', 'r'))
    # progress_bar = tqdm(range(228))
    summ_headers = ['refs_a', 'refs_b', 'refs_comm', 'gen_a', 'gen_b', 'gen_comm']
    for idx, example in enumerate(combined_data):
        if idx < 13:
            continue
        # print()
        print(f"Example {idx}")
        for header in summ_headers:
            # print(header)
            if header in example:
                value = example[header]
                if isinstance(value, list):
                    # print("Value is list")
                    value_list = []
                    for item in value:
                        try:
                            value_list.append(get_paraphrase(item))
                        except Exception as e:
                            open('combined_data.json', 'w').write(json.dumps(combined_data))
                            print(e.message())
                            exit(0)
                    example.update({f"para_{header}": value_list})
                else:
                    # print("Value is not list")
                    try:
                        example.update({f"para_{header}": get_paraphrase(value)})
                    except Exception as e:
                        open('combined_data.json', 'w').write(json.dumps(combined_data))
                        print(e.message())
                        exit(0)
        # progress_bar.update(1)
        time.sleep(10)

open('combined_data.json', 'w').write(json.dumps(combined_data))
#     for model_obj in models:
#         model = model_obj['imported_model'].from_pretrained(model_obj['name']).to(device)
#         print(model.generation_config)
#         model.generation_config.update(kwargs = {'max_length': 512})
#         print(model.generation_config)
#         tokenizer = model_obj['tokenizer'].from_pretrained(model_obj['name'])
#         # generation_config = GenerationConfig.from_pretrained(model_obj['name'], max_new_tokens=512)


#         for idx, example in enumerate(restructured_data):

#             ref_groups = ['refs_a', 'refs_b', 'refs_comm', 'gen_a', 'gen_b', 'gen_comm']
#             for group in ref_groups[:3]:
#                 key = f"para_{group}_{model_obj['short_name']}"
#                 example[key] = [] if key not in example else example[key]
#                 input_sentences = example[group]
#                 for input_sentence in input_sentences:
#                     example[f"para_{group}_{model_obj['short_name']}"].append(generate_sentences(input_sentence, model, tokenizer))

#             if example['split'] != 'train':
#                 for group in ref_groups[3:]:
#                     key = f"para_{group}_{model_obj['short_name']}"
#                     example[key] = [] if key not in example else example[key]
#                     input_sentences = example[group]
#                     for input_sentence in input_sentences:
#                         example[f"para_{group}_{model_obj['short_name']}"].append(generate_sentences(input_sentence, model, tokenizer))

#             restructured_data = restructured_data[:idx] + [example] + restructured_data[idx + 1:]

# models = [
#     {
#         'name': 'eugenesiow/bart-paraphrase',
#         'imported_model': BartForConditionalGeneration,
#         'tokenizer': BartTokenizer,
#         'short_name': 'bart'
#     },
#     {
#         'name': 'prithivida/parrot_paraphraser_on_T5',
#         'imported_model': AutoModelForSeq2SeqLM,
#         'tokenizer': AutoTokenizer,
#         'short_name': 'parrot'
#     }
# ]
# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# def generate_sentences(input_sentence: str, model, tokenizer):
#     global device
#     batch = tokenizer(input_sentence, return_tensors='pt')
#     model.to(device)
#     batch['input_ids'].to(device)
#     # print(device)
#     with torch.no_grad():
#         generated_ids = model.generate(batch['input_ids'], max_new_tokens=512)
#         generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#     return generated_sentence
