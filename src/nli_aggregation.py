import pandas as pd
from typing import Tuple, List, Dict
import json
import wandb
import numpy as np


def get_nli_scores(data_type, component, summ_type, compute = 'triple', alphas=(0, 0, 0), run=None):
    data_path = f"data/results/sentpairs_nli_{component}_{summ_type}_{data_type}.csv"

    if component == 'contrast':
        alpha_ent = alphas[0]
        alpha_neut = alphas[1]
        alpha_cont = alphas[2]
        
        nli_scores = nli_aggregator(
            data_path=data_path,
            preferred_label = 'CONTRADICTION',
            pairwise_aggregation='sum',
            alphas=(alpha_ent,alpha_neut,alpha_cont),
            summ_type=summ_type,
            run=run,
            compute = compute,
            tie_winner = 'CONTRADICTION'
        )
        
    elif component == 'factuality':
        nli_scores = nli_fact_bin_ent(data_path=data_path, summ_type=summ_type)

    records = []
    # retrieve actual summaries for records creation
    dataset_path = f"data/combined_data_{data_type}_split_complete.json"
    dataset = json.load(open(dataset_path, 'r'))
    for example_id, example in enumerate(dataset):
        split = example['split']
        if summ_type == 'gen' and split == 'train':
            continue
        if data_type == 'negation':
            summ_a = example['refs_a'][0] if summ_type == 'ref' else example['gen_a']
            summ_a_neg = example['refs_a_neg'] if summ_type == 'ref' else example['gen_b']
            nli_score_idx = example_id  ## because we only do this for refs
            records.append((summ_type, split, example_id, summ_a, summ_a_neg, nli_scores[nli_score_idx]))
        else:
            summ_a = example['refs_a'][0] if summ_type == 'ref' else example['gen_a']
            summ_b = example['refs_b'][0] if summ_type == 'ref' else example['gen_b']
            summ_comm = example['refs_comm'][0] if summ_type == 'ref' else example['gen_comm']
            nli_score_idx = example_id if summ_type == 'ref' else (example_id - 20)
            if compute == 'triple':
                records.append((summ_type, split, example_id, summ_a, summ_b, summ_comm, nli_scores[nli_score_idx]))
            else:
                records.append((summ_type, split, example_id, summ_a, summ_b, nli_scores[nli_score_idx]))
    return records


def nli_fact_bin_ent(data_path: str = "data/factuality_popular_final.csv", summ_type='ref') -> List[float]:
    df = pd.read_csv(data_path)
    # df['is_ent'] = df.apply(lambda row: 1 if row['1_2_Label']=='ENTAILMENT' or row['2_1_Label']=='ENTAILMENT' else 0, axis = 1)
    df['is_ent'] = df.apply(lambda row: 1 if row['1_2_Label'] == 'ENTAILMENT' else 0, axis=1)
    # display(df)
    pivot_sent = df.pivot_table(values=['is_ent'], index=['Type', 'Sample', 'Sent2_entity', 'Sentence2'], aggfunc='sum',
                                fill_value=0).reset_index().rename(
        mapper={'Sent2_entity': 'summary', 'Sentence2': 'sentence'}, axis=1)
    # display(pivot_sent)

    pivot_sent['is_ent'] = pivot_sent['is_ent'].apply(lambda no_ents: 1 if no_ents > 0 else -1)
    # display(pivot_sent)
    pivot_sent.to_csv("temp.csv")
    pivot_summ_pair = pivot_sent.pivot_table(values=['is_ent'], index=['Type', 'Sample'], aggfunc='mean', fill_value=0)

    # display(pivot_summ_pair)

    # display(pivot_summ_pair[pivot_summ_pair.index.isin(['ref'], level=0)])
    # df.to_csv("pivot_summ_fair.csv")
    return pivot_summ_pair[pivot_summ_pair.index.isin([summ_type], level=0)]['is_ent'].to_list()


# region contrast
mapper_sent1 = {
    'Sentence1': 'sentence',
    'Sentence2': 'other_sentence',
    'Sent1_entity': 'sent_entity',
    'Sent2_entity': 'other_entity',
    '1_2_ent': 'fwd_ent',
    '1_2_neut': 'fwd_neut',
    '1_2_cont': 'fwd_cont',
    '1_2_Label': 'fwd_label',
    '2_1_ent': 'bwd_ent',
    '2_1_neut': 'bwd_neut',
    '2_1_cont': 'bwd_cont',
    '2_1_Label': 'bwd_label',
}
mapper_sent2 = {
    'Sentence2': 'sentence',
    'Sentence1': 'other_sentence',
    'Sent2_entity': 'sent_entity',
    'Sent1_entity': 'other_entity',
    '2_1_ent': 'fwd_ent',
    '2_1_neut': 'fwd_neut',
    '2_1_cont': 'fwd_cont',
    '2_1_Label': 'fwd_label',
    '1_2_ent': 'bwd_ent',
    '1_2_neut': 'bwd_neut',
    '1_2_cont': 'bwd_cont',
    '1_2_Label': 'bwd_label',
}
mapper_summ_sent = {
    'Summary1': 'summ',
    'Sentence2': 'sentence',
    'Summary1_entity': 'summ_entity',
    'Sent2_entity': 'sent_entity',
    '1_2_neut': 'neut',
    '1_2_cont': 'cont',
    '1_2_ent': 'ent',
    '1_2_Label': 'label',
}


def resolve_labels(row):
    if row['fwd_label'] == row['bwd_label']:
        return row['fwd_label'].upper()
    elif 'NEUTRAL' in (labels := [row['fwd_label'].upper(), row['bwd_label'].upper()]):
        labels.remove('NEUTRAL')
        return labels[0].upper()
    else:
        return 'NEUTRAL'

def nli_aggregator(data_path: str, preferred_label, pairwise_aggregation: str = 'sum', alphas: Tuple = (0, 1, 1),
                                summ_type='ref', run=None, compute = 'pair', tie_winner = 'CONTRADICTION') -> List[float]:
    alphas_dict = {
        'ENTAILMENT': (alphas[0], 'ent'),
        'NEUTRAL': (alphas[1], 'neut'),
        'CONTRADICTION': (alphas[2], 'cont'),
    }

    # transform df to sent level instead of comparison level
    df1 = pd.read_csv(data_path)
    df2 = df1.copy(deep=True)
    print(f"Length of initial dataframe = {len(df1)}")
    df1 = df1.rename(mapper=mapper_sent1, axis=1)
    df2 = df2.rename(mapper=mapper_sent2, axis=1)
    df = pd.concat([df1, df2])
    if compute == 'pair':
        df = df[(df['sent_entity'] != 'comm') & (df['other_entity'] != 'comm')]
    print(f"Length of sent level df = {len(df)}")

    for label in ['cont', 'ent', 'neut']:
        if pairwise_aggregation == 'sum':
            df[f'agg_{label}'] = (df[f'fwd_{label}'] + df[f'bwd_{label}']) / 2
        else:
            df[f'agg_ent'] = np.maximum(df[f'fwd_ent'], df[f'bwd_ent'])
            df[f'agg_neut'] = (df[f'fwd_neut'] + df[f'bwd_neut']) / 2
            df[f'agg_cont'] = np.maximum(df[f'fwd_cont'], df[f'bwd_cont'])

    df['resolved_label'] = df.apply(resolve_labels, axis=1)
    for label in alphas_dict:
        df[label] = df['resolved_label'].apply(lambda pair_label: 1 if pair_label == label else 0)
    # log the df to wandb
    table = wandb.Table(dataframe=df)
    run.log({'contrast_sent_level_label_score': table})

    pivot_sent = df.pivot_table(
        values=['fwd_ent', 'fwd_neut', 'fwd_cont', 'bwd_ent', 'bwd_neut', 'bwd_cont', 'agg_ent', 'agg_neut',
                'agg_cont'] + [label for label in alphas_dict],
        index=['Type', 'Sample', 'sent_entity', 'sentence'],
        aggfunc='mean',
        fill_value=0
    ).reset_index()
    print(f"Length of pivot_sent = {len(pivot_sent)}")
    
    if preferred_label == 'none':
        label_names = ['ENTAILMENT', 'CONTRADICTION']
        label_names.remove(tie_winner)
        tie_loser = label_names[0]
        pivot_sent['score'] = pivot_sent.apply(
            lambda row:
            alphas_dict['NEUTRAL'][0] if row['NEUTRAL'] == 1.0 else (
                alphas_dict[tie_winner][0] if row[tie_winner] >= row[tie_loser] else alphas_dict[tie_loser][
                    0]
            ), axis=1
        )
    else:
        label_names = ['ENTAILMENT', 'CONTRADICTION']
        label_names.remove(preferred_label)
        not_preferred_label = label_names[0]
        pivot_sent['score'] = pivot_sent.apply(
            lambda row:
            alphas_dict[preferred_label][0] if row[preferred_label] > 0 else (
                alphas_dict['NEUTRAL'][0] if row['NEUTRAL'] == 1.0 else alphas_dict[not_preferred_label][0]
            ), axis=1
        )
        
    # log the df to wandb
    table = wandb.Table(dataframe=pivot_sent)
    run.log({'contrast_sent_level_label_score': table})

    pivot_summ = pivot_sent.pivot_table(
        values=['score', 'sentence'],
        index=['Type', 'Sample', 'sent_entity'],
        aggfunc={
            'score': ['mean', 'sum'],
            'sentence': ['count'],
        },
        fill_value=0
    )
    pivot_summ.columns = list(map("_".join, pivot_summ.columns))
    print(f"Length of pivot_summ = {len(pivot_summ)}")
    # log the df to wandb
    table = wandb.Table(dataframe=pivot_summ.reset_index())
    run.log({'contrast_summ_level_label_score': table})

    pivot_sample = pivot_sent.pivot_table(
        values=['score', 'sentence'],
        index=['Type', 'Sample'],
        aggfunc={
            'score': ['mean', 'sum'],
            'sentence': ['count'],
        },
        fill_value=0
    )
    pivot_sample.columns = list(map("_".join, pivot_sample.columns))
    print(f"Length of pivot_sample = {len(pivot_sample)}")
    # log the df to wandb
    table = wandb.Table(dataframe=pivot_sample.reset_index())
    run.log({'contrast_sample_level_label_score': table})

    scores = pivot_sample[pivot_sample.index.isin([summ_type], level=0)]['score_mean'].to_list()

    print(f"Mean score = {np.mean(scores)}")
    return scores

# endregion contrast