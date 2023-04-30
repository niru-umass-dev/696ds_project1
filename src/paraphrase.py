import json
import os
from os import listdir
import openai
import time
import rouge
import numpy as np
from typing import List
import nltk
import backoff
nltk.download('punkt')

openai.api_key = os.getenv('OPENAI_API_KEY')

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def _get_paraphrase_(input_sentence: str, max_tokens: int = 250, n: int = 1):
    prompt = f"Paraphrase this: f'{input_sentence}'"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature = 0.5,
        n=n
    )
    return response['choices']


def get_para_output(input_summ: str, control_length: bool, n: int, dummy: bool, tokenizer = None):
  if dummy:
    return "dummy value."
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

def get_para_dataset(orig_dataset_path: str, control_length: bool, n: int, dummy: bool, tokenizer = None, sleep = 0):
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
          new_example.update({f"{header}": value_list})
        else:
          new_example.update({f"{header}": get_para_output(input_summ=value, control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer)})
    paraphrase_data.append(new_example)
    time.sleep(sleep)
  return paraphrase_data

def get_selfpara_dataset(orig_dataset_path: str, src_summ:str, control_length: bool, n: int, dummy: bool, tokenizer = None, sleep = 0):
  # replace "dummy value" with get_paraphrase(item, max_len)
  combined_data = json.load(open(orig_dataset_path, 'r'))
  # progress_bar = tqdm(range(228))
  paraphrase_data = []
  for idx, example in enumerate(combined_data):
    new_example = dict()
    new_example['split'] = example['split']
    new_example['entity_a'] = example['entity_a']
    new_example['entity_b'] = example['entity_b']
    
    new_example['refs_a'] = [example[f"refs_{src_summ}"][0]]
    new_example['refs_b'] = [get_para_output(input_summ=new_example['refs_a'], control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer)]
    new_example['refs_comm'] = [get_para_output(input_summ=new_example['refs_b'], control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer)]
    
    if example['split'] != 'train':
        new_example['gen_a'] = example[f"gen_{src_summ}"]
        new_example['gen_b'] = get_para_output(input_summ=new_example['gen_a'], control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer)
        new_example['gen_comm'] = get_para_output(input_summ=new_example['gen_b'], control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer) 

    print(f"Example {idx}")
    paraphrase_data.append(new_example)
    time.sleep(sleep)
  return paraphrase_data

def get_para_dataset_sent_level(orig_dataset_path: str, control_length: bool, n: int, dummy: bool, tokenizer = None, sleep = 0):
    # replace "dummy value" with get_paraphrase(item, max_len)
    completed_paraphrases = {}
    combined_data = json.load(open(orig_dataset_path, 'r'))
    # progress_bar = tqdm(range(228))
    ref_headers = ['refs_a', 'refs_b', 'refs_comm']
    # gen_headers = ['gen_a', 'gen_b', 'gen_comm']
    paraphrase_data = []
    for idx, example in enumerate(combined_data):
        new_example = dict()
        new_example['split'] = example['split']
        new_example['entity_a'] = example['entity_a']
        new_example['entity_b'] = example['entity_b']
        # print(f"Example {idx}")
        # ONLY REF FIRST SUMMARY
        for header in ref_headers:
            original_summaries = [example[header][0]]
            para_summ_sents_list = []
            for summ_no, summary in enumerate(original_summaries):
                summary_sents = nltk.sent_tokenize(summary)
                para_summ_sents_list.append([get_para_output(input_summ=summary_sent, control_length=control_length, n=n, dummy=dummy, tokenizer=tokenizer) for summary_sent in summary_sents])
            para_summs = [" ".join(para_summ_sents) for para_summ_sents in para_summ_sents_list]
            completed_paraphrases[f"{idx}_{header}_{summ_no}"] = para_summs
            json.dump(completed_paraphrases, open("data/temporary_dataset_files/completed_sent_level_paraphrase_text-davinci-003.json", "w"))
            new_example.update({f"{header}": para_summs})
            
        paraphrase_data.append(new_example)
    return paraphrase_data
