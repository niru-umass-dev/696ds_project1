import wandb
import random
from datetime import datetime
import pytz
import os
import pandas as pd
import json

from src.baseline_metrics import (
    get_evaluator,
    get_scorer,
    get_bs_scores,
    get_ds_scores,
)

# Get timestamp
now = datetime.now()
et_timezone = pytz.timezone('US/Eastern')
now = datetime.now(et_timezone)
datetime_string = now.strftime("%Y%m%d_%H%M%S")


## Hyperparameters
summ_orig = ["ref", "gen"]

# start a new wandb run to track this script
run = wandb.init(
    # set the wandb project where this run will be logged
    project="696ds-contrastive-summ-eval",
    
    # track hyperparameters and run metadata
    config={
        
        "summ_orig": summ_orig, # Are we analyzing reference or generated summaries or both
        "summ_models": ["cococsum"], # if generated, then the model which generates the summaries
        "metrics":["ds", "bs", "nli"], # metrics we are computing
        "nli_models": ["robert-large-mnli"], # model used for nli
        "nli_desid": ["contrast", "factuality"], # which nli desiderata
        "experiments":{ 
            "paraphrase": {
                "paraphrase_model": "text-davinci-003",
                "length_control": "+5%",
                "is_dummy":False, # dummy run to test paraphrasing code without actually consuming openai API
                "summ_set": "all" # cont/all running on only cont summaries or all
            },
            "self_paraphrase": {
                "paraphrase_model": "text-davinci-003",
                "length_control": "+5%",
                "is_dummy":False,
                "summ_set": "all"
            }
        },
        "timestamp": datetime_string,
    },
    notes = "Printing out run ID and name"
)

if not os.path.isdir("data/results"):
    os.mkdir("data/results")

## BASELINES        

combined_data = None

# First Baseline: DS Score
evaluator = None
for summ_type in summ_orig:    
    if not os.path.isfile(f"data/results/ds_{summ_type}.csv"):
        combined_data = json.load(open("data/combined_data.json",'r')) if combined_data is None else combined_data
        evaluator = get_evaluator() if evaluator is None else evaluator
        records = get_ds_scores(combined_data, summ_type, evaluator)
        df = pd.DataFrame.from_records(records, columns = ['summ_type','split','example_id', 'summ_a', 'summ_b', 'summ_comm', 'ds_score'])
        df.to_csv(f"data/results/ds_{summ_type}.csv")
    wandb.save(f"data/results/ds_{summ_type}.csv")
        
# Second Baseline: BertScore
scorer = None
for summ_type in summ_orig:    
    if not os.path.isfile(f"data/results/bs_{summ_type}.csv"):
        combined_data = json.load(open("data/combined_data.json",'r')) if combined_data is None else combined_data
        scorer = get_scorer() if scorer is None else scorer
        records = get_bs_scores(combined_data, summ_type, scorer)
        df = pd.DataFrame.from_records(records, columns = ['summ_type','split','example_id', 'summ_a', 'summ_b', 'summ_comm', 'bs_score'])
        df.to_csv(f"data/results/bs_{summ_type}.csv")
    wandb.save(f"data/results/bs_{summ_type}.csv")

wandb.finish()