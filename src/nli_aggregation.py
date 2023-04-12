import pandas as pd
from typing import Tuple, List, Dict
import json
import wandb
import numpy as np

def get_nli_scores(data_type, component, summ_type, alphas = (0,0,0), run=None):
    data_path = f"data/results/sentpairs_nli_{component}_{summ_type}_{data_type}.csv"
    
    if component == 'contrast':
        alpha_ent = alphas[0]
        alpha_neut = alphas[1]
        alpha_cont = alphas[2]
        nli_scores = nli_contrast_bin_real(data_path=data_path,pairwise_aggregation='mixed',alphas=(alpha_ent,alpha_neut,alpha_cont), summ_type=summ_type, run=run)
        
    elif component == 'factuality':
        nli_scores = nli_fact_bin_ent(data_path=data_path, summ_type=summ_type)
        
    records = []
    # retrieve actual summaries for records creation
    dataset_path = f"data/combined_data_{data_type}.json"
    dataset = json.load(open(dataset_path,'r'))
    for example_id, example in enumerate(dataset):
        split = example['split']
        if summ_type == 'gen' and split == 'train':
            continue
        summ_a = example['refs_a'][0] if summ_type == 'ref' else example['gen_a']
        summ_b = example['refs_b'][0] if summ_type == 'ref' else example['gen_b']
        summ_comm = example['refs_comm'][0] if summ_type == 'ref' else example['gen_comm']
        
        nli_score_idx = example_id if summ_type == 'ref' else (example_id - 20)
        records.append((summ_type, split, example_id, summ_a, summ_b, summ_comm, nli_scores[nli_score_idx]))
    return records

def nli_contrast_real_neut(data_path:str, pairwise_aggregation:str = 'sum', alphas: Tuple = (1,1,1), summ_type='ref', run = None) -> List[float]:
    ## AGGREGATE NLI FOR ORIGINAL
    df = pd.read_csv(data_path)

    for label in ['cont','ent', 'neut']:
        if pairwise_aggregation == 'sum':
            df[f'agg_{label}'] = (df[f'1_2_{label}'] + df[f'2_1_{label}'])/2
        else:
            df[f'agg_{label}'] = np.maximum(df[f'1_2_{label}'], df[f'2_1_{label}'])

    pivot_sent1 = df.pivot_table(values = ['agg_neut'], index = ['Type','Sample','Sent1_entity', 'Sent2_entity', 'Sentence1'], aggfunc = 'mean', fill_value = 0).rename(mapper={'agg_neut':'agg_1_neut'}, axis=1)
    pivot_sent2 = df.pivot_table(values = ['agg_neut'], index = ['Type','Sample','Sent1_entity', 'Sent2_entity', 'Sentence2'], aggfunc = 'mean', fill_value = 0).rename(mapper={'agg_neut':'agg_2_neut'}, axis=1)


    df = df.join(other=pivot_sent1,on=['Type','Sample','Sent1_entity', 'Sent2_entity', 'Sentence1'])
    df = df.join(other=pivot_sent2,on=['Type','Sample','Sent1_entity', 'Sent2_entity', 'Sentence2'])

    # multipliers for each label
    alpha_ent = alphas[0]
    alpha_neut = alphas[1]
    alpha_cont = alphas[2]

    df['ent_final'] = alpha_ent * df['agg_ent']
    df['cont_final'] = alpha_cont * df['agg_cont']
    df['neut_final'] = alpha_neut * ((df['agg_1_neut']*df['1_2_neut']/2+df['agg_2_neut']*df['2_1_neut']/2))

    df['total'] = df['ent_final'] + df['cont_final'] + df['neut_final']

    pivot_summ_pair = df.pivot_table(values = ['total'], index = ['Type','Sample'], aggfunc = 'mean', fill_value = 0)

    return pivot_summ_pair[pivot_summ_pair.index.isin([summ_type], level=0)]['total'].to_list()

def nli_contrast_bin_neut(data_path:str, pairwise_aggregation:str = 'sum', alphas: Tuple = (0,1,1), summ_type='ref', run = None) -> List[float]:
    alpha_ent = alphas[0]
    alpha_neut = alphas[1]
    alpha_cont = alphas[2]
    ## AGGREGATE NLI FOR ORIGINAL
    df = pd.read_csv(data_path)

    for label in ['cont','ent', 'neut']:
        if pairwise_aggregation == 'sum':
            df[f'agg_{label}'] = (df[f'1_2_{label}'] + df[f'2_1_{label}'])/2
        else:
            df[f'agg_{label}'] = np.maximum(df[f'1_2_{label}'], df[f'2_1_{label}'])

    df['ent_final'] = alpha_ent * df['agg_ent']
    df['cont_final'] = alpha_cont * df['agg_cont']
    df['bi_neut'] = df.apply(lambda row: 1 if row['1_2_Label']==row['2_1_Label']=='NEUTRAL' else 0, axis = 1)
    pivot_sent1_entity = df.pivot_table(values = ['agg_neut', 'bi_neut', 'ent_final', 'cont_final'], index = ['Type','Sample', 'Sent1_entity','Sent2_entity', 'Sentence1'], aggfunc = 'mean', fill_value = 0).reset_index().rename(mapper={'Sent1_entity':'sent_entity' ,'Sent2_entity':'other_entity', 'Sentence1':'sentence'}, axis=1)
    pivot_sent2_entity = df.pivot_table(values = ['agg_neut', 'bi_neut', 'ent_final', 'cont_final'], index = ['Type','Sample', 'Sent1_entity', 'Sent2_entity', 'Sentence2'], aggfunc = 'mean', fill_value = 0).reset_index().rename(mapper={'Sent2_entity':'sent_entity' ,'Sent1_entity':'other_entity', 'Sentence2':'sentence'}, axis=1)

    pivot_sent_entity = pd.concat([pivot_sent1_entity, pivot_sent2_entity])

    pivot_sent_entity['bi_neut_binary'] = pivot_sent_entity['bi_neut'].apply(lambda bi_neut: alpha_neut * 1 if bi_neut==1.0 else 0) 

    pivot_summ_pair = pivot_sent_entity.pivot_table(values = ['ent_final','cont_final', 'bi_neut_binary'], index = ['Type','Sample'], aggfunc = 'mean', fill_value = 0).rename(mapper={'bi_neut_binary': 'bi_neut_final'}, axis=1)
    # display(pivot_summ_pair)
    pivot_summ_pair['total'] = pivot_summ_pair['ent_final'] + pivot_summ_pair['cont_final'] + pivot_summ_pair['bi_neut_final']

    # display(pivot_summ_pair)

    # display(pivot_summ_pair[pivot_summ_pair.index.isin(['ref'], level=0)])
    # df.to_csv("pivot_summ_fair.csv")

    return pivot_summ_pair[pivot_summ_pair.index.isin([summ_type], level=0)]['total'].to_list()


def nli_fact_bin_ent(data_path: str = "data/factuality_popular_final.csv", summ_type='ref') -> List[float]:
    df = pd.read_csv(data_path)
    df['is_ent'] = df.apply(lambda row: 1 if row['1_2_Label']=='ENTAILMENT' or row['2_1_Label']=='ENTAILMENT' else 0, axis = 1)
    # display(df)
    pivot_sent = df.pivot_table(values = ['is_ent'], index = ['Type','Sample', 'Sent2_entity','Sentence2'], aggfunc = 'sum', fill_value = 0).reset_index().rename(mapper={'Sent2_entity':'summary' ,'Sentence2':'sentence'}, axis=1)
    # display(pivot_sent)

    pivot_sent['is_ent'] =  pivot_sent['is_ent'].apply(lambda no_ents: 1 if no_ents>0 else -1)
    # display(pivot_sent)
    pivot_sent.to_csv("temp.csv")
    pivot_summ_pair = pivot_sent.pivot_table(values = ['is_ent'], index = ['Type','Sample'], aggfunc = 'mean', fill_value=0)

    # display(pivot_summ_pair)

    # display(pivot_summ_pair[pivot_summ_pair.index.isin(['ref'], level=0)])
    # df.to_csv("pivot_summ_fair.csv")
    return pivot_summ_pair[pivot_summ_pair.index.isin([summ_type], level=0)]['is_ent'].to_list()


def resolve_labels(row):
    if row['fwd_label'] == row['bwd_label']:
        return row['fwd_label']
    elif 'NEUTRAL' in (labels := [row['fwd_label'], row['bwd_label']]):
        labels.remove('NEUTRAL')
        return labels[0]
    else:
        return 'NEUTRAL'


def nli_contrast_bin_all(data_path: str, pairwise_aggregation: str = 'sum', alphas: Tuple = (0, 1, 1),
                         summ_type='ref', run = None) -> List[float]:
    alphas_dict = {
        'ENTAILMENT': alphas[0],
        'NEUTRAL': alphas[1],
        'CONTRADICTION': alphas[2],
    }
    ## AGGREGATE NLI FOR ORIGINAL

    # move around the columns a bit to get a lookup of sent -> comparison
    df1 = pd.read_csv(data_path)
    df2 = df1.copy(deep=True)

    df1 = df1.drop(columns=['Sentence2']).rename(mapper={
        'Sentence1': 'sentence',
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
    }, axis=1)
    df2 = df2.drop(columns=['Sentence1']).rename(mapper={
        'Sentence2': 'sentence',
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
    }, axis=1)

    df = pd.concat([df1, df2])
    for label in ['cont', 'ent', 'neut']:
        if pairwise_aggregation == 'sum':
            df[f'agg_{label}'] = (df[f'fwd_{label}'] + df[f'bwd_{label}']) / 2
        else:
            df[f'agg_{label}'] = np.maximum(df[f'fwd_{label}'], df[f'bwd_{label}'])

    df['label_final'] = df.apply(resolve_labels, axis=1)
    df['label_score'] = df['label_final'].apply(lambda label: alphas_dict[label])

    # log the df to wandb
    table = wandb.Table(dataframe=df)
    run.log({'contrast_sent_level_label_score': table})

    pivot_summ = df.pivot_table(
        values=['label_score', 'sentence'],
        index=['Type', 'Sample', 'sent_entity'],
        aggfunc={
            'label_score': ['mean', 'sum'],
            'sentence': ['count'],
        },
        fill_value=0
    )
    pivot_summ.columns = list(map("_".join, pivot_summ.columns))

    # log the df to wandb
    table = wandb.Table(dataframe=pivot_summ.reset_index())
    run.log({'contrast_summ_level_label_score': table})

    pivot_sample = df.pivot_table(
        values=['label_score', 'sentence'],
        index=['Type', 'Sample'],
        aggfunc={
            'label_score': ['mean', 'sum'],
            'sentence': ['count'],
        },
        fill_value=0
    )
    pivot_sample.columns = list(map("_".join, pivot_sample.columns))

    # log the df to wandb
    table = wandb.Table(dataframe=pivot_sample.reset_index())
    run.log({'contrast_sample_level_label_score': table})
    
    return pivot_sample[pivot_sample.index.isin([summ_type], level=0)]['label_score_mean'].to_list()


def nli_contrast_bin_real(data_path: str, pairwise_aggregation='sum', alphas: Tuple = (0, 1, 1), summ_type='ref', run=None) -> \
        List[float]:
    alphas_dict = {
        'ENTAILMENT': (alphas[0], 'ent'),
        'NEUTRAL': (alphas[1], 'neut'),
        'CONTRADICTION': (alphas[2], 'cont'),
    }
    ## AGGREGATE NLI FOR ORIGINAL

    # move around the columns a bit to get a lookup of sent -> comparison
    df1 = pd.read_csv(data_path)
    df2 = df1.copy(deep=True)

    df1 = df1.drop(columns=['Sentence2']).rename(mapper={
        'Sentence1': 'sentence',
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
    }, axis=1)
    df2 = df2.drop(columns=['Sentence1']).rename(mapper={
        'Sentence2': 'sentence',
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
    }, axis=1)

    df = pd.concat([df1, df2])
    for label in ['cont', 'ent', 'neut']:
        if pairwise_aggregation == 'sum':
            df[f'agg_{label}'] = (df[f'fwd_{label}'] + df[f'bwd_{label}']) / 2
        else:
            df[f'agg_ent'] = np.maximum(df[f'fwd_ent'], df[f'bwd_ent'])
            df[f'agg_neut'] = (df[f'fwd_neut'] + df[f'bwd_neut']) / 2
            df[f'agg_cont'] = np.maximum(df[f'fwd_cont'], df[f'bwd_cont'])

    df['label_final'] = df.apply(resolve_labels, axis=1)
    df['label_score'] = df.apply(
        lambda row: alphas_dict[row['label_final']][0] * row[f"agg_{alphas_dict[row['label_final']][1]}"], axis=1)

    # log the df to wandb
    table = wandb.Table(dataframe=df.reset_index())
    run.log({'contrast_sent_level_label_score': table})

    pivot_summ = df.pivot_table(
        values=['label_score', 'sentence'],
        index=['Type', 'Sample', 'sent_entity'],
        aggfunc={
            'label_score': ['mean', 'sum'],
            'sentence': ['count'],
        },
        fill_value=0
    )
    pivot_summ.columns = list(map("_".join, pivot_summ.columns))

    # log the df to wandb
    table = wandb.Table(dataframe=pivot_summ.reset_index())
    run.log({'contrast_summ_level_label_score': table})

    pivot_sample = df.pivot_table(
        values=['label_score', 'sentence'],
        index=['Type', 'Sample'],
        aggfunc={
            'label_score': ['mean', 'sum'],
            'sentence': ['count'],
        },
        fill_value=0
    )
    pivot_sample.columns = list(map("_".join, pivot_sample.columns))

    # log the df to wandb
    table = wandb.Table(dataframe=pivot_sample.reset_index())
    run.log({'contrast_sample_level_label_score': table})

    return pivot_sample[pivot_sample.index.isin([summ_type], level=0)]['label_score_mean'].to_list()