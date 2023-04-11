import wandb
import random
from datetime import datetime
import pytz
import os
import pandas as pd
import json
from transformers import GPT2TokenizerFast
from paraphrase import (
    get_para_dataset,
    get_selfpara_dataset,
)
import copy
from itertools import product

from src.baseline_metrics import (
    get_evaluator,
    get_scorer,
    get_bs_scores,
    get_ds_scores,
)
from src.nli_modular import SummNLI
from src.nli_aggregation import(
    get_nli_scores
)
from src.experiment_utils import(
    get_centtend_measures
)

# Get timestamp
now = datetime.now()
et_timezone = pytz.timezone('US/Eastern')
now = datetime.now(et_timezone)
datetime_string = now.strftime("%Y%m%d_%H%M%S")


## Hyperparameters
summ_orig = ["ref", "gen"]
metrics = ['ds', 'bs', 'nli']
nli_components = ["contrast", "factuality"]
datasets = ['base', 'paraphrase', 'selfparaphrase']
experiments = {
    "paraphrase": {
        "paraphrase_model": "text-davinci-003",
        "length_control": "+5%",
        "is_dummy":False, # dummy run to test paraphrasing code without actually consuming openai API
        "summ_set": "all" # cont/all running on only cont summaries or all
    },
    "selfparaphrase": {
        "paraphrase_model": "text-davinci-003",
        "length_control": "+5%",
        "is_dummy":False,
        "summ_set": "all"
    }
}
# start a new wandb run to track this script
run = wandb.init(
    # set the wandb project where this run will be logged
    project="696ds-contrastive-summ-eval",
    # track hyperparameters and run metadata
    config={
        
        "summ_orig": summ_orig, # Are we analyzing reference or generated summaries or both
        "summ_models": ["cococsum"], # if generated, then the model which generates the summaries
        "metrics": metrics, # metrics we are computing
        "nli_models": ["robert-large-mnli"], # model used for nli
        "nli_components": nli_components, # which nli desiderata
        "experiments": experiments,
        "timestamp": datetime_string,
    },
    notes = "Trying out NLI aggregation methods"
)
run_folder = f"{datetime_string}_{run.name}"
if not os.path.isdir("data/results"):
    os.mkdir("data/results")

## GENERATE SOURCE FILES FOR EXPERIMENTS IF REQUIRED

## Generate experimental Paraphrase Dataset (Using Length Constraint)
# Currently this paraphrases all 3 reference summaries, make sure to change this behaviour if you don't need all 3
if 'paraphrase' in experiments and not os.path.isfile("data/combined_data_paraphrase.json"):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    paraphrase_dataset = get_para_dataset(
        orig_dataset_path='data/combined_data_base.json',
        control_length=True,
        n=5,
        dummy=False,
        tokenizer=tokenizer,
        sleep = 0 # If you're using a rate-limited API, change this to a non-zero value to space your calls
    )
    json.dump(paraphrase_dataset, open("data/combined_data_paraphrase.json", 'w'))

if 'selfparaphrase' in experiments and not os.path.isfile("data/combined_data_selfparaphrase.json"):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    paraphrase_dataset = get_selfpara_dataset(
        orig_dataset_path='data/combined_data_base.json',
        src_summ = 'a',
        control_length=True,
        n=5,
        dummy=False,
        tokenizer=tokenizer,
        sleep = 0 # If you're using a rate-limited API, change this to a non-zero value to space your calls
    )
    json.dump(paraphrase_dataset, open("data/combined_data_selfparaphrase.json", 'w'))

# Generate Sentence-Pair Level NLI Scores if the files don't exist


for dataset in datasets:
    for summ_type in summ_orig:
        data_path = f"data/combined_data_{dataset}.json"
        
        file_path = f"data/results/sentpairs_nli_contrast_{summ_type}_{dataset}.csv"
        if not os.path.isfile(file_path):
            print(f"Not Found:{file_path}")
            nli_metric = SummNLI(data_path=data_path, data_type=dataset, summ_type=summ_type)
            nli_metric.compute()
        
        file_path = f"data/results/sentpairs_nli_factuality_{summ_type}_{dataset}.csv"
        if dataset == 'base' and not os.path.isfile(file_path):
            print(f"Not Found:{file_path}")
            nli_metric = SummNLI(data_path=data_path, data_type=dataset, summ_type=summ_type)
            nli_metric.compute()

            
            
## Create entity-pair level DS, BS, and NLI Scores for base, paraphrase, and selfparaphrase datasets


exp_metrics = copy.deepcopy(metrics)
exp_metrics.remove('nli')
exp_metrics += [f"nli_{nli_component}" for nli_component in nli_components]
evaluator = None
scorer = None
label_alpha_combs = {
    'ent': (1, 0, 0),
    'neut': (0, 1, 0),
    'cont': (0, 0, 1),
    'agg': (-1, 1, 1)
}

for summ_type in summ_orig:
    for dataset in datasets:
        data_path = f"data/combined_data_{dataset}.json"
        data = json.load(open(data_path,'r'))
        for metric in metrics: 
            if metric == 'ds':
                results_path = f"data/results/{metric}_{summ_type}_{dataset}.csv"
                if not os.path.isfile(results_path):
                    print(f"Not Found:{results_path}")
                    evaluator = get_evaluator() if evaluator is None else evaluator
                    records = get_ds_scores(data, summ_type, evaluator)
                    df = pd.DataFrame.from_records(records, columns = ['summ_type','split','example_id', 'summ_a', 'summ_b', 'summ_comm', f'{metric}_score'])
                    df.to_csv(results_path, index=False)
            elif metric == 'bs':
                results_path = f"data/results/{metric}_{summ_type}_{dataset}.csv"
                if not os.path.isfile(results_path):
                    print(f"Not Found:{results_path}")
                    scorer = get_scorer() if scorer is None else scorer
                    records = get_bs_scores(data, summ_type, scorer)
                    df = pd.DataFrame.from_records(records, columns = ['summ_type','split','example_id', 'summ_a', 'summ_b', 'summ_comm', f'{metric}_score'])
                    df.to_csv(results_path, index=False)
            elif metric == 'nli':
                nli_component ='contrast'

                for label in label_alpha_combs:
                    results_path = f"data/results/{metric}_{nli_component}_{summ_type}_{dataset}_{label}.csv"
                    if not os.path.isfile(results_path):
                        print(f"Not Found:{results_path}")
                        records = get_nli_scores(data_type=dataset, component=nli_component,summ_type=summ_type, alphas=label_alpha_combs[label])
                        df = pd.DataFrame.from_records(records, columns = ['summ_type','split','example_id', 'summ_a', 'summ_b', 'summ_comm', f'{metric}_{nli_component}_{label}_score'])
                        df.to_csv(results_path, index=False)
                        wandb.save(results_path)
                
                nli_component ='factuality'
                if dataset != 'base':
                    continue
                results_path = f"data/results/{metric}_{nli_component}_{summ_type}_{dataset}.csv"
                if not os.path.isfile(results_path):
                    print(f"Not Found:{results_path}")
                    records = get_nli_scores(data_type=dataset, component=nli_component,summ_type=summ_type, alphas=None)
                    df = pd.DataFrame.from_records(records, columns = ['summ_type','split','example_id', 'summ_a', 'summ_b', 'summ_comm', f'{metric}_{nli_component}_score'])
                    df.to_csv(results_path, index=False)
                    wandb.save(results_path)

                    
# EXPERIMENT 1: RESULTS

if not os.path.isdir("data/results/experiment_paraphrase"):
    os.mkdir("data/results/experiment_paraphrase")

headers = []
for summ_type in summ_orig:
    records = []
    for dataset in datasets:
        record = []
        record.append(dataset)
        headers = ['data_type']
        for metric in metrics:
            if metric != 'nli':
                col_name = f"{metric}_score"
                
                # Get base scores
                base_file_path = f"data/results/{metric}_{summ_type}_base.csv"
                base_scores = pd.read_csv(base_file_path)[col_name].to_list()
                
                file_path = f"data/results/{metric}_{summ_type}_{dataset}.csv"
                scores = pd.read_csv(file_path)[col_name].to_list()
                
                mean, std_dev, corr, rank_corr, plt = get_centtend_measures(base_scores, scores)
                plt.savefig(f"data/results/experiment_paraphrase/{summ_type}_{metric}_{dataset}.png")
                
                record.extend([mean, std_dev, corr, rank_corr])
                metric_header_name = metric
                headers.extend([f"mean({metric_header_name})", f"std_dev({metric_header_name})", f"corr({metric_header_name})", f"rank_corr({metric_header_name})"])
            elif metric == 'nli':
                
                nli_component = 'contrast'
                for label in label_alpha_combs.keys():
                    col_name = f"{metric}_{nli_component}_{label}_score"
                    # Get base scores
                    base_file_path = f"data/results/{metric}_{nli_component}_{summ_type}_base_{label}.csv"
                    base_scores = pd.read_csv(base_file_path)[col_name].to_list()
                    
                    file_path = f"data/results/{metric}_{nli_component}_{summ_type}_{dataset}_{label}.csv"
                    scores = pd.read_csv(file_path)[col_name].to_list()
                    
                    mean, std_dev, corr, rank_corr, plt = get_centtend_measures(base_scores, scores)
                    plt.savefig(f"data/results/experiment_paraphrase/{summ_type}_{metric}_{nli_component}_{label}_{dataset}.png")
                    
                    record.extend([mean, std_dev, corr, rank_corr])
                    metric_header_name = f"{metric}_{nli_component}_{label}"
                    headers.extend([f"mean({metric_header_name})", f"std_dev({metric_header_name})", f"corr({metric_header_name})", f"rank_corr({metric_header_name})"])
                
                nli_component = 'factuality'
                col_name = f"{metric}_{nli_component}_score"
                
                # Get base scores
                base_file_path = f"data/results/{metric}_{nli_component}_{summ_type}_base.csv"
                base_scores = pd.read_csv(base_file_path)[col_name].to_list()
                if dataset != 'base':
                    scores = [0] * len(base_scores)
                else:
                    file_path = f"data/results/{metric}_{nli_component}_{summ_type}_{dataset}.csv"
                    scores = pd.read_csv(file_path)[col_name].to_list()
                
                mean, std_dev, corr, rank_corr, plt = get_centtend_measures(base_scores, scores)
                plt.savefig(f"data/results/experiment_paraphrase/{summ_type}_{metric}_{nli_component}_{dataset}.png")
                
                record.extend([mean, std_dev, corr, rank_corr])
                metric_header_name = f"{metric}_{nli_component}"
                headers.extend([f"mean({metric_header_name})", f"std_dev({metric_header_name})", f"corr({metric_header_name})", f"rank_corr({metric_header_name})"])
    
        records.append(record)
    pd.DataFrame(records, columns = headers).to_csv(f"data/results/experiment_paraphrase/{summ_type}_summary.csv", index=False)
            


wandb.finish()
