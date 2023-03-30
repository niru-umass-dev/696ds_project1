"""trivial_baseline_utils.py: Contains utility methods for DS, BertScore, and Paraphrase analysis. __main__ also creates data/combined_data.json"""
import json
import os
from os import listdir
from tqdm.auto import tqdm
import openai
import time
import sys
import rouge
from collections import Counter
import matplotlib.pyplot as plt
from statistics import stdev
from scipy.stats import spearmanr
import numpy as np
from numpy import mean
import bert_score

openai.organization = "org-uqW82WXjsx1QYRwThjvVeQqQ"
openai.api_key = "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"

evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True, stemming=True, ensure_compatibility=True)

SCORER = bert_score.BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)

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
        restructured_data = restructured_data[:split_indices[0]] + restructured_split + restructured_data[
                                                                                        split_indices[-1] + 1:]

    return restructured_data

def stem(x):
    return Counter(evaluator.stem_tokens(evaluator.tokenize_text(x.lower())))


def calc_ds(summ_a, summ_b, summ_comm):
    s_a, s_b, s_c = stem(summ_a), stem(summ_b), stem(summ_comm)
    nr = sum((s_a & s_b).values()) + sum((s_a & s_c).values()) + sum((s_b & s_c).values()) - 2.0 * sum((s_a & s_b & s_c).values())
    dr = sum((s_a | s_b | s_c).values())
    return 1.0 - (nr / dr)

def calc_bertscore(summ_a, summ_b, summ_comm):
  return SCORER.score([summ_a], [summ_b])[2].item()

def compare_orig_para(x, y, orig_type, score_type='DS'):
  print()
  print(f"{score_type} | {orig_type} vs paraphrase".upper())
  # x = ds_gen_orig
  # y = ds_gen_para
  x = [100*value for value in x]
  y = [100*value for value in y]
  plt.scatter(x, y)
  plt.show()

  r = np.corrcoef(x,y)

  rankcorr = spearmanr(x,y)
  

  diffs = [x[i] - y[i] for i in range(len(x))]

  print('\n',"Separate Statistics",'\n')
  print(f"{orig_type}-orig | µ({score_type}) = {np.mean(x):.1f} | σ({score_type}) = {np.std(x):.1f}")
  print(f"{orig_type}-para | µ({score_type}) = {np.mean(y):.1f} | σ({score_type}) = {np.std(y):.1f}")
  print()
  print("Combined Statistics",'\n')
  print(f"Spearman Rank Correlation = {rankcorr}")
  print(f"Correlation = {r[0,1]:.2f}")
  print(f"µ({orig_type}-para) = {np.mean(diffs):.1f} | σ({orig_type}-para) = {np.std(diffs):.1f}", '\n')


def get_ds_scores(dataset_path:str, is_paraphrase: bool, orig_type: str, score_fn_name: str = "calc_ds"):
  score_fn = globals()[score_fn_name]
  assert orig_type in {'gen', 'refs'}
  dataset_type = 'para_' if is_paraphrase else ''

  dataset = json.load(open(dataset_path, 'r'))

  ds_scores = []
  # The range is conditional on what splits are present
  for example_no in (range(20,48) if len(dataset)==48 else range(28)):
    orig_example = dataset[example_no]
    # print(f"Example {example_no}")
    summ_a = orig_example[f'{dataset_type}{orig_type}_a']
    summ_b = orig_example[f'{dataset_type}{orig_type}_b']
    summ_comm = orig_example[f'{dataset_type}{orig_type}_comm']

    if orig_type == 'gen':
      ds_scores.append(score_fn(summ_a, summ_b, summ_comm))
    elif orig_type == 'refs':
      ds_scores.append(mean(
            [
                score_fn(summ_a[idx], summ_b[idx], summ_comm[idx]) for idx in range(len(summ_a))
            ]
        )   
      )
    
  return ds_scores


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


def _get_paraphrase_(input_sentence: str, max_tokens: int = 250, n: int = 1):
    prompt = f"Paraphrase this: f'{input_sentence}'"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        n=n
    )
    return response['choices']


def get_para_output(input_summ: str, control_length: bool, n: int, dummy: bool, tokenizer = None):
  if dummy:
    return "dummy value"
  input_len = len(tokenizer(input_summ)['input_ids'])
  if control_length:
    output_len = round(1.05*input_len)
  else:
    output_len = 250
  
  choices = _get_paraphrase_(input_sentence=input_summ, max_tokens=output_len, n=n)  
  output_candidates = [choice['text'] for choice in choices]
  input_len = len(tokenizer(input_summ)['input_ids'])
  candidate_lens = np.array([len(tokenizer(output_candidate)['input_ids']) for output_candidate in output_candidates])
  min_len_idx = np.argmin(np.abs(candidate_lens - input_len))
  output = output_candidates[min_len_idx]

  return output

def create_para_data_diff(orig_dataset_path: str, control_length: bool, n: int, dummy: bool, tokenizer = None):
  # replace "dummy value" with get_paraphrase(item, max_len)
  combined_data = json.load(open(orig_dataset_path, 'r'))
  # progress_bar = tqdm(range(228))
  ref_headers = ['refs_a', 'refs_b', 'refs_comm']
  gen_headers = ['gen_a', 'gen_b', 'gen_comm']
  paraphrase_data = []
  for idx, example in enumerate(combined_data):
    new_example = dict()
    new_example['split'] = example['split']
    new_example['entity_a'] = example['entity_a']
    new_example['entity_b'] = example['entity_b']
    # print(f"Example {idx}")
    for header in (ref_headers + gen_headers):
      if header in example:
        value = example[header]
        if isinstance(value, list):
          value_list = [get_para_output(input_summ=item, control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer) for item in value]
          new_example.update({f"para_{header}": value_list})
        else:
          new_example.update({f"para_{header}": get_para_output(input_summ=value, control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer)})
    paraphrase_data.append(new_example)
    time.sleep(10)
  return paraphrase_data

def create_para_data_same(orig_dataset_path: str, src_summ:str, control_length: bool, n: int, dummy: bool, tokenizer = None):
  # replace "dummy value" with get_paraphrase(item, max_len)
  combined_data = json.load(open(orig_dataset_path, 'r'))
  # progress_bar = tqdm(range(228))
  paraphrase_data = []
  for idx, example in enumerate(combined_data):
    if src_summ not in example:
      continue
    new_example = dict()
    new_example['split'] = example['split']
    new_example['entity_a'] = example['entity_a']
    new_example['entity_b'] = example['entity_b']

    new_example['gen_a'] = example[src_summ] if src_summ.startswith('gen') else example[src_summ][0]
    new_example['gen_b'] = get_para_output(input_summ=new_example['gen_a'], control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer)
    new_example['gen_comm'] = get_para_output(input_summ=new_example['gen_b'], control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer)
    print(f"Example {idx}")
    paraphrase_data.append(new_example)
    # time.sleep(10)
  return paraphrase_data

if __name__=='__main__':
    print(f"main_path = {sys.argv[0]}")
    project_folder = os.getcwd()[:os.getcwd().find('696ds_project1')+ len('696ds_project1')]
    data_folder = os.path.join(project_folder, 'data')
    combined_data = get_combined_data()
    check_written_dataset(combined_data)
    json.dump(combined_data, open("data/combined_data.json", 'w'))
    
    
    
    # print(f"len of dict = {len(data)}")
    
    # combined_data=get_combined_data()

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

# Code to split extract paraphrase_data__1 from combined_data_paraphrase one-time
# para_data_1 = json.load(open('paraphrase_data_1.json', 'r'))

# for idx in range(len(para_data_1)):
#   item = para_data_1[idx]
#   para_data_1[idx] = {
#       'split': item['split'],
#       'entity_a': item['entity_a'],
#       'entity_b': item['entity_a'],
#       'entity_a_uid': item['entity_a_uid'],
#       'entity_b_uid': item['entity_b_uid'],
#       'para_refs_a': item['para_refs_a'],
#       'para_refs_b': item['para_refs_b'],
#       'para_refs_comm': item['para_refs_comm'],
#   }

#   if item['split'] in ['dev', 'test']:
#     para_data_1[idx].update({
#         'para_gen_a': item['para_gen_a'],
#         'para_gen_b': item['para_gen_b'],
#         'para_gen_comm': item['para_gen_comm'],
#     })

# json.dump(para_data_1, open("paraphrase_data_1.json", 'w'))

## OTHER CODE
# openai.api_key = os.getenv("OPENAI_API_KEY")


# print(openai.Model.list())

# input_sentence = "The hotel has an enjoyable ambience and is great value for money. This location is great if you're looking to access the great fun and action close-by, but with that comes hustle-and-bustle noise from the street the hotel is located on. The rustic rooms here are a little small but really comfortable and clean while the beds are big. There was a thatched roof to the room that provided an airy feel but of course not soundproof. The hotel provides great breakfast to be eaten above the garden or to take away. The hotel does not provide private parking unfortunately."
