import numpy as np
import pandas as pd
import torch
import json
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ROBERTA-LARGE-MNLI 
model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
label_mapping = ['contradiction', 'neutral','entailment']


# CROSS-ENCODER NLI-ROBERTA-BASE
# model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
# tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')
# label_mapping = ['contradiction', 'entailment', 'neutral']

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

data = json.load(open('data/combined_data.json', 'r'))
# data = json.load(open('data/combined_data_masked_spacyner_bertscore.json', 'r'))
# data = json.load(open('data/combined_data_masked_spacyner_spacysim.json', 'r'))


# NLI to get labels and scores
def compute_NLI(sent1, sent2, rev=False):
    
    if rev == True:
        features = tokenizer(sent1,sent2,  padding=True, truncation=True, return_tensors="pt")
        features.to(device) 
    else:    
        features = tokenizer(sent2,sent1,  padding=True, truncation=True, return_tensors="pt")
        features.to(device)

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1).detach().cpu().numpy()]
        
        prob = torch.softmax(scores, dim=1).detach().cpu().numpy()
            
    return labels, prob.tolist()


# HELPER METHOD FOR MAIN FUNCTION HELPFUL CONTRAST
def cont_helper(a_sum, b_sum, comm_sum,sum_type):

    sum_types = []
    sent_p1 = []
    sent_p2 = []
    sent1_ent =[]
    sent2_ent = []

    count =0
    for i in a_sum:
        for j in b_sum:

            sent_p1.append(i)
            sent_p2.append(j)
            sent1_ent.append('a')
            sent2_ent.append('b')
            count+=1

    for i in a_sum:
        for j in comm_sum:

            sent_p1.append(i)
            sent_p2.append(j)
            sent1_ent.append('a')
            sent2_ent.append('comm')
            count+=1

    for i in b_sum:
        for j in comm_sum:

            sent_p1.append(i)
            sent_p2.append(j)
            sent1_ent.append('b')
            sent2_ent.append('comm')
            count+=1

    sum_types += [sum_type] * count

    return sent_p1, sent_p2, sent1_ent, sent2_ent, sum_types


# HELPER METHOD FOR MAIN FUNCTION POPULAR OPINION FACTUAL CONSISTENCY
def fact_helper(source_a, source_b, a_sum, b_sum, comm_sum,sum_type):

    sum_types = []
    sent_p1 = []
    sent_p2 = []
    sent1_source =[]
    sent1_source_ent =[]
    sent2_ent = []

    count =0
    for i in range(len(source_a)):
        for j in source_a[i]:
            for k in a_sum:

                sent_p1.append(j)
                sent_p2.append(k)
                sent1_source.append(i)
                sent1_source_ent.append('a')
                sent2_ent.append('a')
                count+=1

    for i in range(len(source_a)):
        for j in source_a[i]:
            for k in comm_sum:

                sent_p1.append(j)
                sent_p2.append(k)
                sent1_source.append(i)
                sent1_source_ent.append('a')
                sent2_ent.append('comm')
                count+=1

    for i in range(len(source_b)):
        for j in source_b[i]:
            for k in b_sum:

                sent_p1.append(j)
                sent_p2.append(k)
                sent1_source.append(i)
                sent1_source_ent.append('b')
                sent2_ent.append('b')
                count+=1

    for i in range(len(source_b)):
        for j in source_b[i]:
            for k in comm_sum:

                sent_p1.append(j)
                sent_p2.append(k)
                sent1_source.append(i)
                sent1_source_ent.append('b')
                sent2_ent.append('comm')
                count+=1

    sum_types += [sum_type] * count

    return sent_p1, sent_p2, sent1_source, sent1_source_ent, sent2_ent, sum_types

######################################################################################################################

# RUN NLI FOR CONTRAST

cont_labels = []
cont_probs = []

cont_labels_rev = []
cont_probs_rev = []

sent_pair_1 =[]
sent_pair_2 = []
sent1_ent =[]
sent2_ent = []
sample = []
sum_type = []

for d in range(20,len(data)):

    # ref summaries just first reference
    ref_a_sum = nltk.sent_tokenize(data[d]['refs_a'][0])
    ref_b_sum = nltk.sent_tokenize(data[d]['refs_b'][0])
    ref_comm_sum = nltk.sent_tokenize(data[d]['refs_comm'][0])

    # gen summaries
    gen_a_sum = nltk.sent_tokenize(data[d]['gen_a'])
    gen_b_sum = nltk.sent_tokenize(data[d]['gen_b'])
    gen_comm_sum = nltk.sent_tokenize(data[d]['gen_comm'])

    # ref aggregations
    ref_sent_p1, ref_sent_p2, ref_sent1_ent, ref_sent2_ent, ref_count = cont_helper(ref_a_sum, ref_b_sum, ref_comm_sum, 'ref')

    # gen aggregations
    gen_sent_p1, gen_sent_p2, gen_sent1_ent, gen_sent2_ent, gen_count = cont_helper(gen_a_sum, gen_b_sum, gen_comm_sum, 'gen')

    sent_pair_1 += ref_sent_p1 + gen_sent_p1
    sent_pair_2 += ref_sent_p2 + gen_sent_p2
    sent1_ent += ref_sent1_ent + gen_sent1_ent
    sent2_ent += ref_sent2_ent + gen_sent2_ent
    sum_type += ref_count + gen_count

    sample += [d] * len(ref_count + gen_count)

    cont_label, cont_prob = compute_NLI(ref_sent_p1 + gen_sent_p1, ref_sent_p2 + gen_sent_p2)

    cont_label_rev, cont_prob_rev = compute_NLI(ref_sent_p1 + gen_sent_p1, ref_sent_p2 + gen_sent_p2, rev=True)

    cont_labels += cont_label
    cont_probs.append(cont_prob)

    cont_labels_rev += cont_label_rev
    cont_probs_rev.append(cont_prob_rev)

    print(d-20)

    
cont = [j for i in cont_probs for j in np.array(i)[:,label_mapping.index("contradiction")]]
neut = [j for i in cont_probs for j in np.array(i)[:,label_mapping.index("neutral")]]
ent = [j for i in cont_probs for j in np.array(i)[:,label_mapping.index("entailment")]]
cont_rev = [j for i in cont_probs_rev for j in np.array(i)[:,label_mapping.index("contradiction")]]
neut_rev = [j for i in cont_probs_rev for j in np.array(i)[:,label_mapping.index("neutral")]]
ent_rev = [j for i in cont_probs_rev for j in np.array(i)[:,label_mapping.index("entailment")]]


cont_df = pd.DataFrame(
    {'Sentence1': sent_pair_1,
     'Sentence2': sent_pair_2,
     'Sent1_entity': sent1_ent,
     'Sent2_entity': sent2_ent,
     'Sample': sample,
     '1_2_neut':  neut,
     '1_2_cont': cont,
     '1_2_ent': ent,
     '1_2_Label': cont_labels,
     '2_1_neut': neut_rev,
     '2_1_cont': cont_rev,
     '2_1_ent': ent_rev, 
     '2_1_Label': cont_labels_rev,
     'Type': sum_type
    })

##############################################################################################################

# RUN NLI FOR FACTUAL CONSISTENCY AND POPULAR OPINION

fact_labels = []
fact_probs = []
sent_pair_1 =[]
sent_pair_2 = []
sent1_source =[]
sent1_source_ent =[]
sent2_ent = []
sample = []
sum_type = []

for d in range(20,len(data)):

    # source reviews 
    source_a = [nltk.sent_tokenize(i) for i in data[d]['source_reviews_a']]
    source_b = [nltk.sent_tokenize(i) for i in data[d]['source_reviews_b']]

    # ref summaries just first reference
    ref_a_sum = nltk.sent_tokenize(data[d]['refs_a'][0])
    ref_b_sum = nltk.sent_tokenize(data[d]['refs_b'][0])
    ref_comm_sum = nltk.sent_tokenize(data[d]['refs_comm'][0])

    # ref aggregations
    ref_sent_p1, ref_sent_p2, ref_sent1_source, ref_sent1_source_ent,ref_sent2_ent, ref_count = fact_helper(source_a, source_b, ref_a_sum, ref_b_sum, ref_comm_sum, 'ref')

    sent_pair_1 += ref_sent_p1 
    sent_pair_2 += ref_sent_p2 
    sent1_source += ref_sent1_source 
    sent1_source_ent += ref_sent1_source_ent
    sent2_ent += ref_sent2_ent 
    sum_type += ref_count 

    sample += [d] * len(ref_count)

    fact_label, fact_prob = compute_NLI(ref_sent_p1,ref_sent_p2)

    fact_labels += fact_label
    fact_probs.append(fact_prob)

    print(d-20)
    

fact_labels_gen = []
fact_probs_gen = []
sent_pair_1_gen =[]
sent_pair_2_gen = []
sent1_source_gen =[]
sent1_source_ent_gen =[]
sent2_ent_gen = []
sample_gen = []
sum_type_gen = []

for d in range(20,len(data)):

    # source reviews 
    source_a = [nltk.sent_tokenize(i) for i in data[d]['source_reviews_a']]
    source_b = [nltk.sent_tokenize(i) for i in data[d]['source_reviews_b']]

    # gen summaries
    gen_a_sum = nltk.sent_tokenize(data[d]['gen_a'])
    gen_b_sum = nltk.sent_tokenize(data[d]['gen_b'])
    gen_comm_sum = nltk.sent_tokenize(data[d]['gen_comm'])

    # gen aggregations
    gen_sent_p1, gen_sent_p2, gen_sent1_source, gen_sent1_source_ent, gen_sent2_ent, gen_count = fact_helper(source_a, source_b, gen_a_sum, gen_b_sum, gen_comm_sum, 'gen')

    sent_pair_1_gen +=  gen_sent_p1
    sent_pair_2_gen +=  gen_sent_p2
    sent1_source_gen +=  gen_sent1_source
    sent1_source_ent_gen +=  gen_sent1_source_ent
    sent2_ent_gen +=  gen_sent2_ent
    sum_type_gen += gen_count

    sample_gen += [d] * len(gen_count)

    fact_label_gen, fact_prob_gen = compute_NLI(gen_sent_p1, gen_sent_p2)

    fact_labels_gen += fact_label_gen
    fact_probs_gen.append(fact_prob_gen)

    print(d-20)
    
cont = [j for i in fact_probs for j in np.array(i)[:,label_mapping.index("contradiction")]]
neut = [j for i in fact_probs for j in np.array(i)[:,label_mapping.index("neutral")]]
ent = [j for i in fact_probs for j in np.array(i)[:,label_mapping.index("entailment")]]
cont_gen = [j for i in fact_probs_gen for j in np.array(i)[:,label_mapping.index("contradiction")]]
neut_gen = [j for i in fact_probs_gen for j in np.array(i)[:,label_mapping.index("neutral")]]
ent_gen = [j for i in fact_probs_gen for j in np.array(i)[:,label_mapping.index("entailment")]]


fact_pop_df = pd.DataFrame(
    {'Sentence1': sent_pair_1 + sent_pair_1_gen,
     'Sentence2': sent_pair_2 + sent_pair_2_gen,
     'Sample': sample + sample_gen,
     'Sent1_source': sent1_source + sent1_source_gen,
     'Sent1_source_entity': sent1_source_ent + sent1_source_ent_gen,
     'Sent2_entity': sent2_ent + sent2_ent_gen,
     '1_2_neut':  neut + neut_gen,
     '1_2_cont': cont + cont_gen,
     '1_2_ent': ent + ent_gen,
     'Label': fact_labels +fact_labels_gen,
     'Type': sum_type + sum_type_gen
    })


####################################################################################################################

# SAVE CSV FILES

cont_df.to_csv('contrast.csv', index = None, header=True) 
fact_pop_df.to_csv('factuality_popular.csv', index = None, header=True) 

# cont_df.to_csv('contrast_mask_bertscore.csv', index = None, header=True) 
# fact_pop_df.to_csv('factuality_popular_mask_bertscore.csv', index = None, header=True) 

# cont_df.to_csv('contrast_mask_spacysim.csv', index = None, header=True) 
# fact_pop_df.to_csv('factuality_popular_mask_spacysim.csv', index = None, header=True) 

