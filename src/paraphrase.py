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
from transformers import GPT2TokenizerFast
from src.synonyms_prompt import SYNONYMS_PROMPT
from nltk.corpus import stopwords
import string
nltk.download('punkt')

openai.api_key = os.getenv('OPENAI_API_KEY')


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def _get_paraphrase_(input_sentence: str, max_tokens: int = 512, n: int = 1):
    prompt = f"Paraphrase this: f'{input_sentence}'"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.5,
        n=n
    )
    return response['choices']


def get_para_output(input_summ: str, control_length: bool, n: int, dummy: bool, tokenizer=None):
    if dummy:
        return "dummy value."
    input_len = len(tokenizer(input_summ)['input_ids'])
    if control_length:
        output_len = round(1.05 * input_len)
    else:
        output_len = 250

    choices = _get_paraphrase_(input_sentence=input_summ, max_tokens=output_len, n=n)
    output_candidates = [choice['text'] for choice in choices]
    input_len = len(tokenizer(input_summ)['input_ids'])
    candidate_lens = np.array([len(tokenizer(output_candidate)['input_ids']) for output_candidate in output_candidates])
    min_len_idx = np.argmin(np.abs(candidate_lens - input_len))
    output = output_candidates[min_len_idx]

    return output


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def _get_paraphrase_synonyms(input_sentence: str, max_tokens: int = 512, n: int = 1):
    prompt = SYNONYMS_PROMPT.format(input_sentence)
    # print("INPUT")
    # print(prompt)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=1.0,
        n=n
    )
    return response['choices']


def get_para_output_synonyms(input_summ: str, control_length: bool, n: int, dummy: bool, tokenizer=None):
    if dummy:
        return "dummy value."
    input_len = len(tokenizer(input_summ)['input_ids'])
    if control_length:
        output_len = round(1.05 * input_len)
    else:
        output_len = 250

    choices = _get_paraphrase_synonyms(input_sentence=input_summ, max_tokens=output_len, n=n)
    output_candidates = [choice['text'].strip("\n ") for choice in choices]
    input_len = len(tokenizer(input_summ)['input_ids'])
    candidate_lens = np.array([len(tokenizer(output_candidate)['input_ids']) for output_candidate in output_candidates])
    min_len_idx = np.argmin(np.abs(candidate_lens - input_len))
    output = output_candidates[min_len_idx]
    # print("OUTPUT")
    # print(output)
    return output


def get_para_dataset(orig_dataset_path: str, control_length: bool, n: int, dummy: bool, tokenizer=None, sleep=0):
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
                    value_list = [get_para_output(input_summ=item, control_length=control_length, n=n, dummy=dummy,
                                                  tokenizer=tokenizer) for item in value]
                    new_example.update({f"{header}": value_list})
                else:
                    new_example.update({f"{header}": get_para_output(input_summ=value, control_length=control_length,
                                                                     n=n, dummy=dummy, tokenizer=tokenizer)})
        paraphrase_data.append(new_example)
        time.sleep(sleep)
    return paraphrase_data


def get_selfpara_dataset(orig_dataset_path: str, src_summ: str, control_length: bool, n: int, dummy: bool,
                         tokenizer=None, sleep=0):
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
        new_example['refs_b'] = [
            get_para_output(input_summ=new_example['refs_a'], control_length=control_length, n=n, dummy=dummy,
                            tokenizer=tokenizer)]
        new_example['refs_comm'] = [
            get_para_output(input_summ=new_example['refs_b'], control_length=control_length, n=n, dummy=dummy,
                            tokenizer=tokenizer)]

        if example['split'] != 'train':
            new_example['gen_a'] = example[f"gen_{src_summ}"]
            new_example['gen_b'] = get_para_output(input_summ=new_example['gen_a'], control_length=control_length, n=n,
                                                   dummy=dummy, tokenizer=tokenizer)
            new_example['gen_comm'] = get_para_output(input_summ=new_example['gen_b'], control_length=control_length,
                                                      n=n, dummy=dummy, tokenizer=tokenizer)

        print(f"Example {idx}")
        paraphrase_data.append(new_example)
        time.sleep(sleep)
    return paraphrase_data


def get_para_dataset_sent_level(orig_dataset_path: str, control_length: bool, n: int, dummy: bool, tokenizer=None,
                                sleep=0):
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
                para_summ_sents_list.append([get_para_output(input_summ=summary_sent, control_length=control_length,
                                                             n=n, dummy=dummy, tokenizer=tokenizer) for summary_sent in
                                             summary_sents])
            para_summs = [" ".join(para_summ_sents) for para_summ_sents in para_summ_sents_list]
            completed_paraphrases[f"{idx}_{header}_{summ_no}"] = para_summs
            json.dump(completed_paraphrases,
                      open("data/temporary_dataset_files/completed_sent_level_paraphrase_text-davinci-003.json", "w"))
            new_example.update({f"{header}": para_summs})

        paraphrase_data.append(new_example)
    return paraphrase_data


def get_selfpara_dataset_sent_level(orig_dataset_path: str, temp_save_path: str, control_length: bool, n: int,
                                    dummy: bool, tokenizer=None, sleep=0):
    # replace "dummy value" with get_paraphrase(item, max_len)
    completed_paraphrases = {} if not os.path.isfile(temp_save_path) else json.load(open(temp_save_path, 'r'))
    combined_data = json.load(open(orig_dataset_path, 'r'))
    # progress_bar = tqdm(range(228))
    src_header_name = 'refs_a'
    tgt_header_names = ['refs_b', 'refs_comm']
    # gen_headers = ['gen_a', 'gen_b', 'gen_comm']
    paraphrase_data = []
    for idx, example in enumerate(combined_data):
        new_example = dict()
        new_example['split'] = example['split']
        new_example['entity_a'] = example['entity_a']
        new_example['entity_b'] = example['entity_b']
        original_summaries = [example[src_header_name][0]]
        new_example.update({f"{src_header_name}": original_summaries})
        for header in tgt_header_names:
            para_summ_sents_list = []
            for summ_no, summary in enumerate(original_summaries):
                summary_sents = nltk.sent_tokenize(summary)
                para_summ_sents_list.append([get_para_output(input_summ=summary_sent, control_length=control_length,
                                                             n=n, dummy=dummy, tokenizer=tokenizer) for summary_sent in
                                             summary_sents])
            para_summs = [" ".join(para_summ_sents) for para_summ_sents in para_summ_sents_list]
            completed_paraphrases[f"{idx}_{header}_{summ_no}"] = para_summs
            json.dump(completed_paraphrases, open(temp_save_path, "w"))
            new_example.update({f"{header}": para_summs})
            original_summaries = para_summs

        paraphrase_data.append(new_example)
    return paraphrase_data


def get_para_dataset_synonyms(orig_dataset_path: str, temp_save_path:str, control_length: bool, n: int, dummy: bool, tokenizer=None,
                              sleep=0):
    completed_paraphrases = {} if not os.path.isfile(temp_save_path) else json.load(open(temp_save_path, 'r'))
    # replace "dummy value" with get_paraphrase(item, max_len)
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
                para_summ_sents_list.append([get_para_output_synonyms(input_summ=summary_sent,
                                                                      control_length=control_length, n=n, dummy=dummy,
                                                                      tokenizer=tokenizer) for summary_sent in
                                             summary_sents])
            para_summs = [" ".join(para_summ_sents) for para_summ_sents in para_summ_sents_list]
            completed_paraphrases[f"{idx}_{header}_{summ_no}"] = para_summs
            json.dump(completed_paraphrases, open(temp_save_path, "w"))
            new_example.update({f"{header}": para_summs})

        paraphrase_data.append(new_example)
    return paraphrase_data


if __name__ == '__main__':
    source_dataset_path = "data/combined_data_base.json"
    save_folder_name = "paraphrase_synonyms"
    temp_file_name = "combined_data_paraphrase_synonyms_temp.json"
    final_file_name = "combined_data_paraphrase_synonyms.json"

    save_folder_path = os.path.join('data/temporary_dataset_files', save_folder_name)
    if not os.path.isdir(save_folder_path):
        os.mkdir(save_folder_path)

    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    temp_file_path = os.path.join(save_folder_path, temp_file_name)
    paraphrase_data = get_para_dataset_synonyms(
        orig_dataset_path=source_dataset_path,
        temp_save_path=temp_file_path,
        control_length=False,
        n=1,
        dummy=False,
        tokenizer=gpt2_tokenizer,
        sleep=0
    )

    final_file_path = os.path.join(save_folder_path, final_file_name)
    json.dump(paraphrase_data, open(final_file_path, "w"))
    
    
    # nltk.download('stopwords')
    # stopwordset = set(stopwords.words('english'))
    # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # in_sent = "Around the holidays season, the hotel was well decorated for X-mas with lovely Palm trees and Christmas lights. This hotel is so close to the beach and is perfectly located for peace and quiet."
    # paragraph = "It was overpriced here at this hotel and the cleaning standards were only okay. This hotel isn't in a great location because it was outside of the old town, which is where some good attractions are. Whilst somewhat nice to be away from the main strip it could have been closer to the main attractions. Some of the staff were rude and unhelpful. You can eat your breakfast at the outside restaurant when staying at this hotel, which is really enjoyable. This is a great stay and you can ask for a reduced rate if you are a Florida resident. The pool is really worth checking out, as well at the hot tub."
    # paragraph_depunct = paragraph.translate(str.maketrans('', '', string.punctuation))
    # word_set = set(nltk.word_tokenize(paragraph_depunct))
    # content_wordset = word_set.difference(stopwordset)
    # content_words = list(content_wordset)
    # # print(content_words)
    # out_sent = get_para_output_synonyms(in_sent, False, 1, False, gpt2_tokenizer)
    # print("OUTPUT")
    # print(out_sent)